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


class HierarchicalModel(DiffusionModel):
    """Simple hierarchical wrapper to combine group-level models."""

    def __init__(self, model: DiffusionModel, groups: Sequence[str]):
        args = [model, groups]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁ__init____mutmut_orig(self, model: DiffusionModel, groups: Sequence[str]):
        self.template = model
        self.groups = list(groups)
        self._params: dict[str, float] = {}

    def xǁHierarchicalModelǁ__init____mutmut_1(self, model: DiffusionModel, groups: Sequence[str]):
        self.template = None
        self.groups = list(groups)
        self._params: dict[str, float] = {}

    def xǁHierarchicalModelǁ__init____mutmut_2(self, model: DiffusionModel, groups: Sequence[str]):
        self.template = model
        self.groups = None
        self._params: dict[str, float] = {}

    def xǁHierarchicalModelǁ__init____mutmut_3(self, model: DiffusionModel, groups: Sequence[str]):
        self.template = model
        self.groups = list(None)
        self._params: dict[str, float] = {}

    def xǁHierarchicalModelǁ__init____mutmut_4(self, model: DiffusionModel, groups: Sequence[str]):
        self.template = model
        self.groups = list(groups)
        self._params: dict[str, float] = None
    
    xǁHierarchicalModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁ__init____mutmut_1': xǁHierarchicalModelǁ__init____mutmut_1, 
        'xǁHierarchicalModelǁ__init____mutmut_2': xǁHierarchicalModelǁ__init____mutmut_2, 
        'xǁHierarchicalModelǁ__init____mutmut_3': xǁHierarchicalModelǁ__init____mutmut_3, 
        'xǁHierarchicalModelǁ__init____mutmut_4': xǁHierarchicalModelǁ__init____mutmut_4
    }
    xǁHierarchicalModelǁ__init____mutmut_orig.__name__ = 'xǁHierarchicalModelǁ__init__'

    # ------------------------------------------------------------------
    # DiffusionModel API helpers
    # ------------------------------------------------------------------
    @property
    def param_names(self) -> Sequence[str]:
        names: list[str] = [f"global_{p}" for p in self.template.param_names]
        for g in self.groups:
            for p in self.template.param_names:
                names.append(f"{g}_{p}")
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = None
        base = self.template.initial_guesses(t, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = None
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(None, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, None)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, )
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = None
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = None
        return guesses

    def xǁHierarchicalModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 1.0
        return guesses
    
    xǁHierarchicalModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁinitial_guesses__mutmut_1': xǁHierarchicalModelǁinitial_guesses__mutmut_1, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_2': xǁHierarchicalModelǁinitial_guesses__mutmut_2, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_3': xǁHierarchicalModelǁinitial_guesses__mutmut_3, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_4': xǁHierarchicalModelǁinitial_guesses__mutmut_4, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_5': xǁHierarchicalModelǁinitial_guesses__mutmut_5, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_6': xǁHierarchicalModelǁinitial_guesses__mutmut_6, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_7': xǁHierarchicalModelǁinitial_guesses__mutmut_7, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_8': xǁHierarchicalModelǁinitial_guesses__mutmut_8, 
        'xǁHierarchicalModelǁinitial_guesses__mutmut_9': xǁHierarchicalModelǁinitial_guesses__mutmut_9
    }
    xǁHierarchicalModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁHierarchicalModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(t, y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = None
        base = self.template.bounds(t, y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = None
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(None, y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(t, None)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(t, )
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(t, y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = None
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def xǁHierarchicalModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(t, y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = None
        return bounds
    
    xǁHierarchicalModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁbounds__mutmut_1': xǁHierarchicalModelǁbounds__mutmut_1, 
        'xǁHierarchicalModelǁbounds__mutmut_2': xǁHierarchicalModelǁbounds__mutmut_2, 
        'xǁHierarchicalModelǁbounds__mutmut_3': xǁHierarchicalModelǁbounds__mutmut_3, 
        'xǁHierarchicalModelǁbounds__mutmut_4': xǁHierarchicalModelǁbounds__mutmut_4, 
        'xǁHierarchicalModelǁbounds__mutmut_5': xǁHierarchicalModelǁbounds__mutmut_5, 
        'xǁHierarchicalModelǁbounds__mutmut_6': xǁHierarchicalModelǁbounds__mutmut_6, 
        'xǁHierarchicalModelǁbounds__mutmut_7': xǁHierarchicalModelǁbounds__mutmut_7, 
        'xǁHierarchicalModelǁbounds__mutmut_8': xǁHierarchicalModelǁbounds__mutmut_8
    }
    xǁHierarchicalModelǁbounds__mutmut_orig.__name__ = 'xǁHierarchicalModelǁbounds'

    def fit(self, t: Sequence[float], y):
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁfit__mutmut_orig(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_1(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = None
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_2(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = None

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_3(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = None
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_4(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = None
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_5(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(None, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_6(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, None, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_7(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, None)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_8(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_9(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_10(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, )
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_11(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = None
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_12(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = None
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_13(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = None
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_14(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(None)
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_15(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(None))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_16(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(None)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_17(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = None
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_18(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(None, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_19(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, None, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_20(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, None)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_21(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_22(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_23(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, )
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_24(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = None
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_25(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = None

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_26(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 1.0

        self._params = params
        return self

    def xǁHierarchicalModelǁfit__mutmut_27(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = None
        return self
    
    xǁHierarchicalModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁfit__mutmut_1': xǁHierarchicalModelǁfit__mutmut_1, 
        'xǁHierarchicalModelǁfit__mutmut_2': xǁHierarchicalModelǁfit__mutmut_2, 
        'xǁHierarchicalModelǁfit__mutmut_3': xǁHierarchicalModelǁfit__mutmut_3, 
        'xǁHierarchicalModelǁfit__mutmut_4': xǁHierarchicalModelǁfit__mutmut_4, 
        'xǁHierarchicalModelǁfit__mutmut_5': xǁHierarchicalModelǁfit__mutmut_5, 
        'xǁHierarchicalModelǁfit__mutmut_6': xǁHierarchicalModelǁfit__mutmut_6, 
        'xǁHierarchicalModelǁfit__mutmut_7': xǁHierarchicalModelǁfit__mutmut_7, 
        'xǁHierarchicalModelǁfit__mutmut_8': xǁHierarchicalModelǁfit__mutmut_8, 
        'xǁHierarchicalModelǁfit__mutmut_9': xǁHierarchicalModelǁfit__mutmut_9, 
        'xǁHierarchicalModelǁfit__mutmut_10': xǁHierarchicalModelǁfit__mutmut_10, 
        'xǁHierarchicalModelǁfit__mutmut_11': xǁHierarchicalModelǁfit__mutmut_11, 
        'xǁHierarchicalModelǁfit__mutmut_12': xǁHierarchicalModelǁfit__mutmut_12, 
        'xǁHierarchicalModelǁfit__mutmut_13': xǁHierarchicalModelǁfit__mutmut_13, 
        'xǁHierarchicalModelǁfit__mutmut_14': xǁHierarchicalModelǁfit__mutmut_14, 
        'xǁHierarchicalModelǁfit__mutmut_15': xǁHierarchicalModelǁfit__mutmut_15, 
        'xǁHierarchicalModelǁfit__mutmut_16': xǁHierarchicalModelǁfit__mutmut_16, 
        'xǁHierarchicalModelǁfit__mutmut_17': xǁHierarchicalModelǁfit__mutmut_17, 
        'xǁHierarchicalModelǁfit__mutmut_18': xǁHierarchicalModelǁfit__mutmut_18, 
        'xǁHierarchicalModelǁfit__mutmut_19': xǁHierarchicalModelǁfit__mutmut_19, 
        'xǁHierarchicalModelǁfit__mutmut_20': xǁHierarchicalModelǁfit__mutmut_20, 
        'xǁHierarchicalModelǁfit__mutmut_21': xǁHierarchicalModelǁfit__mutmut_21, 
        'xǁHierarchicalModelǁfit__mutmut_22': xǁHierarchicalModelǁfit__mutmut_22, 
        'xǁHierarchicalModelǁfit__mutmut_23': xǁHierarchicalModelǁfit__mutmut_23, 
        'xǁHierarchicalModelǁfit__mutmut_24': xǁHierarchicalModelǁfit__mutmut_24, 
        'xǁHierarchicalModelǁfit__mutmut_25': xǁHierarchicalModelǁfit__mutmut_25, 
        'xǁHierarchicalModelǁfit__mutmut_26': xǁHierarchicalModelǁfit__mutmut_26, 
        'xǁHierarchicalModelǁfit__mutmut_27': xǁHierarchicalModelǁfit__mutmut_27
    }
    xǁHierarchicalModelǁfit__mutmut_orig.__name__ = 'xǁHierarchicalModelǁfit'

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁpredict__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError(None)

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = None
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(None)
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = None
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = None
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = None
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(None, 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", None)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", )
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 1.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = None
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(None, 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", None)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", )
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 1.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = None
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base - adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = None
            total += B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total = B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total -= B.array(m.predict(t, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(None)
        return total

    def xǁHierarchicalModelǁpredict__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(None, covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, None))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(covariates))
        return total

    def xǁHierarchicalModelǁpredict__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, ))
        return total
    
    xǁHierarchicalModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁpredict__mutmut_1': xǁHierarchicalModelǁpredict__mutmut_1, 
        'xǁHierarchicalModelǁpredict__mutmut_2': xǁHierarchicalModelǁpredict__mutmut_2, 
        'xǁHierarchicalModelǁpredict__mutmut_3': xǁHierarchicalModelǁpredict__mutmut_3, 
        'xǁHierarchicalModelǁpredict__mutmut_4': xǁHierarchicalModelǁpredict__mutmut_4, 
        'xǁHierarchicalModelǁpredict__mutmut_5': xǁHierarchicalModelǁpredict__mutmut_5, 
        'xǁHierarchicalModelǁpredict__mutmut_6': xǁHierarchicalModelǁpredict__mutmut_6, 
        'xǁHierarchicalModelǁpredict__mutmut_7': xǁHierarchicalModelǁpredict__mutmut_7, 
        'xǁHierarchicalModelǁpredict__mutmut_8': xǁHierarchicalModelǁpredict__mutmut_8, 
        'xǁHierarchicalModelǁpredict__mutmut_9': xǁHierarchicalModelǁpredict__mutmut_9, 
        'xǁHierarchicalModelǁpredict__mutmut_10': xǁHierarchicalModelǁpredict__mutmut_10, 
        'xǁHierarchicalModelǁpredict__mutmut_11': xǁHierarchicalModelǁpredict__mutmut_11, 
        'xǁHierarchicalModelǁpredict__mutmut_12': xǁHierarchicalModelǁpredict__mutmut_12, 
        'xǁHierarchicalModelǁpredict__mutmut_13': xǁHierarchicalModelǁpredict__mutmut_13, 
        'xǁHierarchicalModelǁpredict__mutmut_14': xǁHierarchicalModelǁpredict__mutmut_14, 
        'xǁHierarchicalModelǁpredict__mutmut_15': xǁHierarchicalModelǁpredict__mutmut_15, 
        'xǁHierarchicalModelǁpredict__mutmut_16': xǁHierarchicalModelǁpredict__mutmut_16, 
        'xǁHierarchicalModelǁpredict__mutmut_17': xǁHierarchicalModelǁpredict__mutmut_17, 
        'xǁHierarchicalModelǁpredict__mutmut_18': xǁHierarchicalModelǁpredict__mutmut_18, 
        'xǁHierarchicalModelǁpredict__mutmut_19': xǁHierarchicalModelǁpredict__mutmut_19, 
        'xǁHierarchicalModelǁpredict__mutmut_20': xǁHierarchicalModelǁpredict__mutmut_20, 
        'xǁHierarchicalModelǁpredict__mutmut_21': xǁHierarchicalModelǁpredict__mutmut_21, 
        'xǁHierarchicalModelǁpredict__mutmut_22': xǁHierarchicalModelǁpredict__mutmut_22, 
        'xǁHierarchicalModelǁpredict__mutmut_23': xǁHierarchicalModelǁpredict__mutmut_23, 
        'xǁHierarchicalModelǁpredict__mutmut_24': xǁHierarchicalModelǁpredict__mutmut_24, 
        'xǁHierarchicalModelǁpredict__mutmut_25': xǁHierarchicalModelǁpredict__mutmut_25, 
        'xǁHierarchicalModelǁpredict__mutmut_26': xǁHierarchicalModelǁpredict__mutmut_26, 
        'xǁHierarchicalModelǁpredict__mutmut_27': xǁHierarchicalModelǁpredict__mutmut_27, 
        'xǁHierarchicalModelǁpredict__mutmut_28': xǁHierarchicalModelǁpredict__mutmut_28, 
        'xǁHierarchicalModelǁpredict__mutmut_29': xǁHierarchicalModelǁpredict__mutmut_29, 
        'xǁHierarchicalModelǁpredict__mutmut_30': xǁHierarchicalModelǁpredict__mutmut_30, 
        'xǁHierarchicalModelǁpredict__mutmut_31': xǁHierarchicalModelǁpredict__mutmut_31
    }
    xǁHierarchicalModelǁpredict__mutmut_orig.__name__ = 'xǁHierarchicalModelǁpredict'

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
    ):
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = None
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(None, covariates)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, None)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(covariates)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, )
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = None
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(None, n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), n=None)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), )
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(None), n=1)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), n=2)
        return np.concatenate([[rates[0]], rates])

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate(None)

    def xǁHierarchicalModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[1]], rates])
    
    xǁHierarchicalModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_1': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_1, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_2': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_2, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_3': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_3, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_4': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_4, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_5': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_5, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_6': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_6, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_7': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_7, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_8': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_8, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_9': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_9, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_10': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_10, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_11': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_11, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_12': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_12, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_13': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_13, 
        'xǁHierarchicalModelǁpredict_adoption_rate__mutmut_14': xǁHierarchicalModelǁpredict_adoption_rate__mutmut_14
    }
    xǁHierarchicalModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁHierarchicalModelǁpredict_adoption_rate'

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        args = [t, y, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁHierarchicalModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁHierarchicalModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁHierarchicalModelǁscore__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = None
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(None, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, None)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, )
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = None
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum(None)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) * 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) + y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(None) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 3)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) + B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(None) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(None))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁHierarchicalModelǁscore__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁHierarchicalModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁHierarchicalModelǁscore__mutmut_1': xǁHierarchicalModelǁscore__mutmut_1, 
        'xǁHierarchicalModelǁscore__mutmut_2': xǁHierarchicalModelǁscore__mutmut_2, 
        'xǁHierarchicalModelǁscore__mutmut_3': xǁHierarchicalModelǁscore__mutmut_3, 
        'xǁHierarchicalModelǁscore__mutmut_4': xǁHierarchicalModelǁscore__mutmut_4, 
        'xǁHierarchicalModelǁscore__mutmut_5': xǁHierarchicalModelǁscore__mutmut_5, 
        'xǁHierarchicalModelǁscore__mutmut_6': xǁHierarchicalModelǁscore__mutmut_6, 
        'xǁHierarchicalModelǁscore__mutmut_7': xǁHierarchicalModelǁscore__mutmut_7, 
        'xǁHierarchicalModelǁscore__mutmut_8': xǁHierarchicalModelǁscore__mutmut_8, 
        'xǁHierarchicalModelǁscore__mutmut_9': xǁHierarchicalModelǁscore__mutmut_9, 
        'xǁHierarchicalModelǁscore__mutmut_10': xǁHierarchicalModelǁscore__mutmut_10, 
        'xǁHierarchicalModelǁscore__mutmut_11': xǁHierarchicalModelǁscore__mutmut_11, 
        'xǁHierarchicalModelǁscore__mutmut_12': xǁHierarchicalModelǁscore__mutmut_12, 
        'xǁHierarchicalModelǁscore__mutmut_13': xǁHierarchicalModelǁscore__mutmut_13, 
        'xǁHierarchicalModelǁscore__mutmut_14': xǁHierarchicalModelǁscore__mutmut_14, 
        'xǁHierarchicalModelǁscore__mutmut_15': xǁHierarchicalModelǁscore__mutmut_15, 
        'xǁHierarchicalModelǁscore__mutmut_16': xǁHierarchicalModelǁscore__mutmut_16, 
        'xǁHierarchicalModelǁscore__mutmut_17': xǁHierarchicalModelǁscore__mutmut_17, 
        'xǁHierarchicalModelǁscore__mutmut_18': xǁHierarchicalModelǁscore__mutmut_18, 
        'xǁHierarchicalModelǁscore__mutmut_19': xǁHierarchicalModelǁscore__mutmut_19, 
        'xǁHierarchicalModelǁscore__mutmut_20': xǁHierarchicalModelǁscore__mutmut_20, 
        'xǁHierarchicalModelǁscore__mutmut_21': xǁHierarchicalModelǁscore__mutmut_21, 
        'xǁHierarchicalModelǁscore__mutmut_22': xǁHierarchicalModelǁscore__mutmut_22, 
        'xǁHierarchicalModelǁscore__mutmut_23': xǁHierarchicalModelǁscore__mutmut_23, 
        'xǁHierarchicalModelǁscore__mutmut_24': xǁHierarchicalModelǁscore__mutmut_24, 
        'xǁHierarchicalModelǁscore__mutmut_25': xǁHierarchicalModelǁscore__mutmut_25, 
        'xǁHierarchicalModelǁscore__mutmut_26': xǁHierarchicalModelǁscore__mutmut_26, 
        'xǁHierarchicalModelǁscore__mutmut_27': xǁHierarchicalModelǁscore__mutmut_27, 
        'xǁHierarchicalModelǁscore__mutmut_28': xǁHierarchicalModelǁscore__mutmut_28, 
        'xǁHierarchicalModelǁscore__mutmut_29': xǁHierarchicalModelǁscore__mutmut_29, 
        'xǁHierarchicalModelǁscore__mutmut_30': xǁHierarchicalModelǁscore__mutmut_30
    }
    xǁHierarchicalModelǁscore__mutmut_orig.__name__ = 'xǁHierarchicalModelǁscore'

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        """HierarchicalModel has no direct differential equation."""
        raise NotImplementedError
