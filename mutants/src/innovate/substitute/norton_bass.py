from collections.abc import Sequence

import numpy as np

from innovate.backend import current_backend as B

from ..base import DiffusionModel
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


class NortonBassModel(DiffusionModel):
    """Norton-Bass Model for successive generations of technologies."""

    def __init__(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        args = [n_generations, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁ__init____mutmut_orig(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_1(
        self,
        n_generations: int = 2,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_2(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations <= 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_3(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 2:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_4(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError(None)
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_5(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("XXNumber of generations must be at least 1.XX")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_6(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_7(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("NUMBER OF GENERATIONS MUST BE AT LEAST 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_8(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = None
        self._params: dict[str, float] = {}
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_9(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = None
        self.covariates = covariates or []

    def xǁNortonBassModelǁ__init____mutmut_10(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = None

    def xǁNortonBassModelǁ__init____mutmut_11(
        self,
        n_generations: int = 1,
        covariates: Sequence[str] | None = None,
    ):
        if n_generations < 1:
            raise ValueError("Number of generations must be at least 1.")
        self.n_generations = n_generations
        self._params: dict[str, float] = {}
        self.covariates = covariates and []
    
    xǁNortonBassModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁ__init____mutmut_1': xǁNortonBassModelǁ__init____mutmut_1, 
        'xǁNortonBassModelǁ__init____mutmut_2': xǁNortonBassModelǁ__init____mutmut_2, 
        'xǁNortonBassModelǁ__init____mutmut_3': xǁNortonBassModelǁ__init____mutmut_3, 
        'xǁNortonBassModelǁ__init____mutmut_4': xǁNortonBassModelǁ__init____mutmut_4, 
        'xǁNortonBassModelǁ__init____mutmut_5': xǁNortonBassModelǁ__init____mutmut_5, 
        'xǁNortonBassModelǁ__init____mutmut_6': xǁNortonBassModelǁ__init____mutmut_6, 
        'xǁNortonBassModelǁ__init____mutmut_7': xǁNortonBassModelǁ__init____mutmut_7, 
        'xǁNortonBassModelǁ__init____mutmut_8': xǁNortonBassModelǁ__init____mutmut_8, 
        'xǁNortonBassModelǁ__init____mutmut_9': xǁNortonBassModelǁ__init____mutmut_9, 
        'xǁNortonBassModelǁ__init____mutmut_10': xǁNortonBassModelǁ__init____mutmut_10, 
        'xǁNortonBassModelǁ__init____mutmut_11': xǁNortonBassModelǁ__init____mutmut_11
    }
    xǁNortonBassModelǁ__init____mutmut_orig.__name__ = 'xǁNortonBassModelǁ__init__'

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for i in range(self.n_generations):
            names.extend([f"p{i + 1}", f"q{i + 1}", f"m{i + 1}"])

        for cov in self.covariates:
            for i in range(self.n_generations):
                names.extend(
                    [f"beta_p{i + 1}_{cov}", f"beta_q{i + 1}_{cov}", f"beta_m{i + 1}_{cov}"],
                )
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁinitial_guesses__mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁinitial_guesses__mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁinitial_guesses__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = None
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = None
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(None)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(None):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = None
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i - 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 2}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 1.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = None
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i - 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 2}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 1.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = None

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i - 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 2}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y * self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(None):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = None
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i - 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 2}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 1.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = None
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i - 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 2}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 1.0
                guesses[f"beta_m{i + 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = None
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i - 1}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 2}_{cov}"] = 0.0
        return guesses

    def xǁNortonBassModelǁinitial_guesses__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            guesses[f"p{i + 1}"] = 0.001
            guesses[f"q{i + 1}"] = 0.1
            guesses[f"m{i + 1}"] = max_y / self.n_generations

        for cov in self.covariates:
            for i in range(self.n_generations):
                guesses[f"beta_p{i + 1}_{cov}"] = 0.0
                guesses[f"beta_q{i + 1}_{cov}"] = 0.0
                guesses[f"beta_m{i + 1}_{cov}"] = 1.0
        return guesses
    
    xǁNortonBassModelǁinitial_guesses__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁinitial_guesses__mutmut_1': xǁNortonBassModelǁinitial_guesses__mutmut_1, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_2': xǁNortonBassModelǁinitial_guesses__mutmut_2, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_3': xǁNortonBassModelǁinitial_guesses__mutmut_3, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_4': xǁNortonBassModelǁinitial_guesses__mutmut_4, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_5': xǁNortonBassModelǁinitial_guesses__mutmut_5, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_6': xǁNortonBassModelǁinitial_guesses__mutmut_6, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_7': xǁNortonBassModelǁinitial_guesses__mutmut_7, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_8': xǁNortonBassModelǁinitial_guesses__mutmut_8, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_9': xǁNortonBassModelǁinitial_guesses__mutmut_9, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_10': xǁNortonBassModelǁinitial_guesses__mutmut_10, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_11': xǁNortonBassModelǁinitial_guesses__mutmut_11, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_12': xǁNortonBassModelǁinitial_guesses__mutmut_12, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_13': xǁNortonBassModelǁinitial_guesses__mutmut_13, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_14': xǁNortonBassModelǁinitial_guesses__mutmut_14, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_15': xǁNortonBassModelǁinitial_guesses__mutmut_15, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_16': xǁNortonBassModelǁinitial_guesses__mutmut_16, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_17': xǁNortonBassModelǁinitial_guesses__mutmut_17, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_18': xǁNortonBassModelǁinitial_guesses__mutmut_18, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_19': xǁNortonBassModelǁinitial_guesses__mutmut_19, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_20': xǁNortonBassModelǁinitial_guesses__mutmut_20, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_21': xǁNortonBassModelǁinitial_guesses__mutmut_21, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_22': xǁNortonBassModelǁinitial_guesses__mutmut_22, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_23': xǁNortonBassModelǁinitial_guesses__mutmut_23, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_24': xǁNortonBassModelǁinitial_guesses__mutmut_24, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_25': xǁNortonBassModelǁinitial_guesses__mutmut_25, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_26': xǁNortonBassModelǁinitial_guesses__mutmut_26, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_27': xǁNortonBassModelǁinitial_guesses__mutmut_27, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_28': xǁNortonBassModelǁinitial_guesses__mutmut_28, 
        'xǁNortonBassModelǁinitial_guesses__mutmut_29': xǁNortonBassModelǁinitial_guesses__mutmut_29
    }
    xǁNortonBassModelǁinitial_guesses__mutmut_orig.__name__ = 'xǁNortonBassModelǁinitial_guesses'

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁbounds__mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁbounds__mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁbounds__mutmut_orig(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_1(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = None
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_2(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = None
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_3(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(None)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_4(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(None):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_5(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = None
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_6(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i - 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_7(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 2}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_8(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1.000001, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_9(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 1.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_10(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = None
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_11(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i - 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_12(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 2}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_13(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1.000001, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_14(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 2.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_15(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = None

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_16(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i - 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_17(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 2}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_18(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (1, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_19(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y / 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_20(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 3)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_21(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(None):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_22(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = None
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_23(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i - 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_24(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 2}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_25(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (+np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_26(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = None
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_27(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i - 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_28(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 2}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_29(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (+np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_30(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = None
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_31(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i - 1}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_32(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 2}_{cov}"] = (-np.inf, np.inf)
        return bounds

    def xǁNortonBassModelǁbounds__mutmut_33(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {}
        max_y = B.max(y)
        for i in range(self.n_generations):
            bounds[f"p{i + 1}"] = (1e-6, 0.1)
            bounds[f"q{i + 1}"] = (1e-6, 1.0)
            bounds[f"m{i + 1}"] = (0, max_y * 2)

        for cov in self.covariates:
            for i in range(self.n_generations):
                bounds[f"beta_p{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_q{i + 1}_{cov}"] = (-np.inf, np.inf)
                bounds[f"beta_m{i + 1}_{cov}"] = (+np.inf, np.inf)
        return bounds
    
    xǁNortonBassModelǁbounds__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁbounds__mutmut_1': xǁNortonBassModelǁbounds__mutmut_1, 
        'xǁNortonBassModelǁbounds__mutmut_2': xǁNortonBassModelǁbounds__mutmut_2, 
        'xǁNortonBassModelǁbounds__mutmut_3': xǁNortonBassModelǁbounds__mutmut_3, 
        'xǁNortonBassModelǁbounds__mutmut_4': xǁNortonBassModelǁbounds__mutmut_4, 
        'xǁNortonBassModelǁbounds__mutmut_5': xǁNortonBassModelǁbounds__mutmut_5, 
        'xǁNortonBassModelǁbounds__mutmut_6': xǁNortonBassModelǁbounds__mutmut_6, 
        'xǁNortonBassModelǁbounds__mutmut_7': xǁNortonBassModelǁbounds__mutmut_7, 
        'xǁNortonBassModelǁbounds__mutmut_8': xǁNortonBassModelǁbounds__mutmut_8, 
        'xǁNortonBassModelǁbounds__mutmut_9': xǁNortonBassModelǁbounds__mutmut_9, 
        'xǁNortonBassModelǁbounds__mutmut_10': xǁNortonBassModelǁbounds__mutmut_10, 
        'xǁNortonBassModelǁbounds__mutmut_11': xǁNortonBassModelǁbounds__mutmut_11, 
        'xǁNortonBassModelǁbounds__mutmut_12': xǁNortonBassModelǁbounds__mutmut_12, 
        'xǁNortonBassModelǁbounds__mutmut_13': xǁNortonBassModelǁbounds__mutmut_13, 
        'xǁNortonBassModelǁbounds__mutmut_14': xǁNortonBassModelǁbounds__mutmut_14, 
        'xǁNortonBassModelǁbounds__mutmut_15': xǁNortonBassModelǁbounds__mutmut_15, 
        'xǁNortonBassModelǁbounds__mutmut_16': xǁNortonBassModelǁbounds__mutmut_16, 
        'xǁNortonBassModelǁbounds__mutmut_17': xǁNortonBassModelǁbounds__mutmut_17, 
        'xǁNortonBassModelǁbounds__mutmut_18': xǁNortonBassModelǁbounds__mutmut_18, 
        'xǁNortonBassModelǁbounds__mutmut_19': xǁNortonBassModelǁbounds__mutmut_19, 
        'xǁNortonBassModelǁbounds__mutmut_20': xǁNortonBassModelǁbounds__mutmut_20, 
        'xǁNortonBassModelǁbounds__mutmut_21': xǁNortonBassModelǁbounds__mutmut_21, 
        'xǁNortonBassModelǁbounds__mutmut_22': xǁNortonBassModelǁbounds__mutmut_22, 
        'xǁNortonBassModelǁbounds__mutmut_23': xǁNortonBassModelǁbounds__mutmut_23, 
        'xǁNortonBassModelǁbounds__mutmut_24': xǁNortonBassModelǁbounds__mutmut_24, 
        'xǁNortonBassModelǁbounds__mutmut_25': xǁNortonBassModelǁbounds__mutmut_25, 
        'xǁNortonBassModelǁbounds__mutmut_26': xǁNortonBassModelǁbounds__mutmut_26, 
        'xǁNortonBassModelǁbounds__mutmut_27': xǁNortonBassModelǁbounds__mutmut_27, 
        'xǁNortonBassModelǁbounds__mutmut_28': xǁNortonBassModelǁbounds__mutmut_28, 
        'xǁNortonBassModelǁbounds__mutmut_29': xǁNortonBassModelǁbounds__mutmut_29, 
        'xǁNortonBassModelǁbounds__mutmut_30': xǁNortonBassModelǁbounds__mutmut_30, 
        'xǁNortonBassModelǁbounds__mutmut_31': xǁNortonBassModelǁbounds__mutmut_31, 
        'xǁNortonBassModelǁbounds__mutmut_32': xǁNortonBassModelǁbounds__mutmut_32, 
        'xǁNortonBassModelǁbounds__mutmut_33': xǁNortonBassModelǁbounds__mutmut_33
    }
    xǁNortonBassModelǁbounds__mutmut_orig.__name__ = 'xǁNortonBassModelǁbounds'

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁpredict__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError(None)

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = None

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(None)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = None

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[1] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1.000001

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = None

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_12(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(None, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, None, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, None, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, None, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, None)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, )

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = None
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            None,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            None,
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            None,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=None,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method=None,
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_28(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_29(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_30(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_31(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_32(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_33(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[1], t[-1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_34(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[+1]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_35(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-2]),
            y0,
            t_eval=t,
            method="LSODA",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_36(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="XXLSODAXX",
        )
        return sol.y.T

    def xǁNortonBassModelǁpredict__mutmut_37(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        from scipy.integrate import solve_ivp

        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y0 = B.zeros(self.n_generations)

        # Set a small initial value for the first generation to kickstart the diffusion
        y0[0] = 1e-6

        params = [self._params[name] for name in self.param_names]

        def ode_func(t, y):
            return self.differential_equation(t, y, params, covariates, t)

        sol = solve_ivp(
            ode_func,
            (t[0], t[-1]),
            y0,
            t_eval=t,
            method="lsoda",
        )
        return sol.y.T
    
    xǁNortonBassModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁpredict__mutmut_1': xǁNortonBassModelǁpredict__mutmut_1, 
        'xǁNortonBassModelǁpredict__mutmut_2': xǁNortonBassModelǁpredict__mutmut_2, 
        'xǁNortonBassModelǁpredict__mutmut_3': xǁNortonBassModelǁpredict__mutmut_3, 
        'xǁNortonBassModelǁpredict__mutmut_4': xǁNortonBassModelǁpredict__mutmut_4, 
        'xǁNortonBassModelǁpredict__mutmut_5': xǁNortonBassModelǁpredict__mutmut_5, 
        'xǁNortonBassModelǁpredict__mutmut_6': xǁNortonBassModelǁpredict__mutmut_6, 
        'xǁNortonBassModelǁpredict__mutmut_7': xǁNortonBassModelǁpredict__mutmut_7, 
        'xǁNortonBassModelǁpredict__mutmut_8': xǁNortonBassModelǁpredict__mutmut_8, 
        'xǁNortonBassModelǁpredict__mutmut_9': xǁNortonBassModelǁpredict__mutmut_9, 
        'xǁNortonBassModelǁpredict__mutmut_10': xǁNortonBassModelǁpredict__mutmut_10, 
        'xǁNortonBassModelǁpredict__mutmut_11': xǁNortonBassModelǁpredict__mutmut_11, 
        'xǁNortonBassModelǁpredict__mutmut_12': xǁNortonBassModelǁpredict__mutmut_12, 
        'xǁNortonBassModelǁpredict__mutmut_13': xǁNortonBassModelǁpredict__mutmut_13, 
        'xǁNortonBassModelǁpredict__mutmut_14': xǁNortonBassModelǁpredict__mutmut_14, 
        'xǁNortonBassModelǁpredict__mutmut_15': xǁNortonBassModelǁpredict__mutmut_15, 
        'xǁNortonBassModelǁpredict__mutmut_16': xǁNortonBassModelǁpredict__mutmut_16, 
        'xǁNortonBassModelǁpredict__mutmut_17': xǁNortonBassModelǁpredict__mutmut_17, 
        'xǁNortonBassModelǁpredict__mutmut_18': xǁNortonBassModelǁpredict__mutmut_18, 
        'xǁNortonBassModelǁpredict__mutmut_19': xǁNortonBassModelǁpredict__mutmut_19, 
        'xǁNortonBassModelǁpredict__mutmut_20': xǁNortonBassModelǁpredict__mutmut_20, 
        'xǁNortonBassModelǁpredict__mutmut_21': xǁNortonBassModelǁpredict__mutmut_21, 
        'xǁNortonBassModelǁpredict__mutmut_22': xǁNortonBassModelǁpredict__mutmut_22, 
        'xǁNortonBassModelǁpredict__mutmut_23': xǁNortonBassModelǁpredict__mutmut_23, 
        'xǁNortonBassModelǁpredict__mutmut_24': xǁNortonBassModelǁpredict__mutmut_24, 
        'xǁNortonBassModelǁpredict__mutmut_25': xǁNortonBassModelǁpredict__mutmut_25, 
        'xǁNortonBassModelǁpredict__mutmut_26': xǁNortonBassModelǁpredict__mutmut_26, 
        'xǁNortonBassModelǁpredict__mutmut_27': xǁNortonBassModelǁpredict__mutmut_27, 
        'xǁNortonBassModelǁpredict__mutmut_28': xǁNortonBassModelǁpredict__mutmut_28, 
        'xǁNortonBassModelǁpredict__mutmut_29': xǁNortonBassModelǁpredict__mutmut_29, 
        'xǁNortonBassModelǁpredict__mutmut_30': xǁNortonBassModelǁpredict__mutmut_30, 
        'xǁNortonBassModelǁpredict__mutmut_31': xǁNortonBassModelǁpredict__mutmut_31, 
        'xǁNortonBassModelǁpredict__mutmut_32': xǁNortonBassModelǁpredict__mutmut_32, 
        'xǁNortonBassModelǁpredict__mutmut_33': xǁNortonBassModelǁpredict__mutmut_33, 
        'xǁNortonBassModelǁpredict__mutmut_34': xǁNortonBassModelǁpredict__mutmut_34, 
        'xǁNortonBassModelǁpredict__mutmut_35': xǁNortonBassModelǁpredict__mutmut_35, 
        'xǁNortonBassModelǁpredict__mutmut_36': xǁNortonBassModelǁpredict__mutmut_36, 
        'xǁNortonBassModelǁpredict__mutmut_37': xǁNortonBassModelǁpredict__mutmut_37
    }
    xǁNortonBassModelǁpredict__mutmut_orig.__name__ = 'xǁNortonBassModelǁpredict'

    def differential_equation(self, t, y, params, covariates, t_eval):
        args = [t, y, params, covariates, t_eval]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁdifferential_equation__mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁdifferential_equation__mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁdifferential_equation__mutmut_orig(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_1(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = None
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_2(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = None
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_3(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 / self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_4(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 3 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_5(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = None

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_6(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 / self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_7(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[3 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_8(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 / self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_9(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 4 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_10(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = None
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_11(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(None)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_12(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = None
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_13(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(None)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_14(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = None

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_15(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(None)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_16(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = None
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_17(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 / self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_18(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 4 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_19(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = None
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_20(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(None, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_21(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, None, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_22(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, None)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_23(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_24(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_25(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, )
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_26(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(None):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_27(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] = params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_28(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] -= params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_29(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] / cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_30(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] = params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_31(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] -= params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_32(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] / cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_33(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx - 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_34(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 2] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_35(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] = params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_36(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] -= params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_37(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] / cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_38(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx - 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_39(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 3] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_40(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx = 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_41(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx -= 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_42(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 4

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_43(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = None

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_44(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(None)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_45(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(None):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_46(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = None
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_47(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 1
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_48(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i <= self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_49(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations + 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_50(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 2:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_51(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = None
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_52(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(None)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_53(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = None

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_54(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(None)

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_55(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i - 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_56(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 2 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_57(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = None

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_58(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) / (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_59(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] - q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_60(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] * m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_61(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] / y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_62(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] + cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_63(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] + y[i] - cannibalization) if m_t[i] > 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_64(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] >= 0 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_65(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 1 else 0

        return dydt

    def xǁNortonBassModelǁdifferential_equation__mutmut_66(self, t, y, params, covariates, t_eval):
        """System of differential equations for the Norton-Bass model."""
        p_base = params[: self.n_generations]
        q_base = params[self.n_generations : 2 * self.n_generations]
        m_base = params[2 * self.n_generations : 3 * self.n_generations]

        p_t = B.array(p_base)
        q_t = B.array(q_base)
        m_t = B.array(m_base)

        if covariates:
            param_idx = 3 * self.n_generations
            for cov_name, cov_values in covariates.items():
                cov_val_t = np.interp(t, t_eval, cov_values)
                for i in range(self.n_generations):
                    p_t[i] += params[param_idx] * cov_val_t
                    q_t[i] += params[param_idx + 1] * cov_val_t
                    m_t[i] += params[param_idx + 2] * cov_val_t
                    param_idx += 3

        dydt = B.zeros_like(y)

        for i in range(self.n_generations):
            # Cannibalization term
            cannibalization = 0
            if i < self.n_generations - 1:
                # Ensure y is treated as a 1D array for summation
                y_flat = B.ravel(y)
                cannibalization = B.sum(y_flat[i + 1 :])

            # Bass diffusion equation for each generation
            dydt[i] = (p_t[i] + q_t[i] * y[i] / m_t[i]) * (m_t[i] - y[i] - cannibalization) if m_t[i] > 0 else 1

        return dydt
    
    xǁNortonBassModelǁdifferential_equation__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁdifferential_equation__mutmut_1': xǁNortonBassModelǁdifferential_equation__mutmut_1, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_2': xǁNortonBassModelǁdifferential_equation__mutmut_2, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_3': xǁNortonBassModelǁdifferential_equation__mutmut_3, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_4': xǁNortonBassModelǁdifferential_equation__mutmut_4, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_5': xǁNortonBassModelǁdifferential_equation__mutmut_5, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_6': xǁNortonBassModelǁdifferential_equation__mutmut_6, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_7': xǁNortonBassModelǁdifferential_equation__mutmut_7, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_8': xǁNortonBassModelǁdifferential_equation__mutmut_8, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_9': xǁNortonBassModelǁdifferential_equation__mutmut_9, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_10': xǁNortonBassModelǁdifferential_equation__mutmut_10, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_11': xǁNortonBassModelǁdifferential_equation__mutmut_11, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_12': xǁNortonBassModelǁdifferential_equation__mutmut_12, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_13': xǁNortonBassModelǁdifferential_equation__mutmut_13, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_14': xǁNortonBassModelǁdifferential_equation__mutmut_14, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_15': xǁNortonBassModelǁdifferential_equation__mutmut_15, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_16': xǁNortonBassModelǁdifferential_equation__mutmut_16, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_17': xǁNortonBassModelǁdifferential_equation__mutmut_17, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_18': xǁNortonBassModelǁdifferential_equation__mutmut_18, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_19': xǁNortonBassModelǁdifferential_equation__mutmut_19, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_20': xǁNortonBassModelǁdifferential_equation__mutmut_20, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_21': xǁNortonBassModelǁdifferential_equation__mutmut_21, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_22': xǁNortonBassModelǁdifferential_equation__mutmut_22, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_23': xǁNortonBassModelǁdifferential_equation__mutmut_23, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_24': xǁNortonBassModelǁdifferential_equation__mutmut_24, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_25': xǁNortonBassModelǁdifferential_equation__mutmut_25, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_26': xǁNortonBassModelǁdifferential_equation__mutmut_26, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_27': xǁNortonBassModelǁdifferential_equation__mutmut_27, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_28': xǁNortonBassModelǁdifferential_equation__mutmut_28, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_29': xǁNortonBassModelǁdifferential_equation__mutmut_29, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_30': xǁNortonBassModelǁdifferential_equation__mutmut_30, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_31': xǁNortonBassModelǁdifferential_equation__mutmut_31, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_32': xǁNortonBassModelǁdifferential_equation__mutmut_32, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_33': xǁNortonBassModelǁdifferential_equation__mutmut_33, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_34': xǁNortonBassModelǁdifferential_equation__mutmut_34, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_35': xǁNortonBassModelǁdifferential_equation__mutmut_35, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_36': xǁNortonBassModelǁdifferential_equation__mutmut_36, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_37': xǁNortonBassModelǁdifferential_equation__mutmut_37, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_38': xǁNortonBassModelǁdifferential_equation__mutmut_38, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_39': xǁNortonBassModelǁdifferential_equation__mutmut_39, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_40': xǁNortonBassModelǁdifferential_equation__mutmut_40, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_41': xǁNortonBassModelǁdifferential_equation__mutmut_41, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_42': xǁNortonBassModelǁdifferential_equation__mutmut_42, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_43': xǁNortonBassModelǁdifferential_equation__mutmut_43, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_44': xǁNortonBassModelǁdifferential_equation__mutmut_44, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_45': xǁNortonBassModelǁdifferential_equation__mutmut_45, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_46': xǁNortonBassModelǁdifferential_equation__mutmut_46, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_47': xǁNortonBassModelǁdifferential_equation__mutmut_47, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_48': xǁNortonBassModelǁdifferential_equation__mutmut_48, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_49': xǁNortonBassModelǁdifferential_equation__mutmut_49, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_50': xǁNortonBassModelǁdifferential_equation__mutmut_50, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_51': xǁNortonBassModelǁdifferential_equation__mutmut_51, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_52': xǁNortonBassModelǁdifferential_equation__mutmut_52, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_53': xǁNortonBassModelǁdifferential_equation__mutmut_53, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_54': xǁNortonBassModelǁdifferential_equation__mutmut_54, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_55': xǁNortonBassModelǁdifferential_equation__mutmut_55, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_56': xǁNortonBassModelǁdifferential_equation__mutmut_56, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_57': xǁNortonBassModelǁdifferential_equation__mutmut_57, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_58': xǁNortonBassModelǁdifferential_equation__mutmut_58, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_59': xǁNortonBassModelǁdifferential_equation__mutmut_59, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_60': xǁNortonBassModelǁdifferential_equation__mutmut_60, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_61': xǁNortonBassModelǁdifferential_equation__mutmut_61, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_62': xǁNortonBassModelǁdifferential_equation__mutmut_62, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_63': xǁNortonBassModelǁdifferential_equation__mutmut_63, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_64': xǁNortonBassModelǁdifferential_equation__mutmut_64, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_65': xǁNortonBassModelǁdifferential_equation__mutmut_65, 
        'xǁNortonBassModelǁdifferential_equation__mutmut_66': xǁNortonBassModelǁdifferential_equation__mutmut_66
    }
    xǁNortonBassModelǁdifferential_equation__mutmut_orig.__name__ = 'xǁNortonBassModelǁdifferential_equation'

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        args = [t, y, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁscore__mutmut_orig(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_1(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_2(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError(None)
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_3(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_4(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_5(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_6(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = None

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_7(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(None, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_8(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, None)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_9(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_10(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, )

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_11(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) != 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_12(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 2:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_13(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = None

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_14(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(None, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_15(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, None)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_16(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_17(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, )

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_18(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(+1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_19(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-2, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_20(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 2)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_21(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = None
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_22(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum(None)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_23(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) * 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_24(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y + y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_25(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 3)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_26(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_27(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_28(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_29(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y + B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_30(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(None, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_31(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_32(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_33(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, )) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_34(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=1)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_35(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_36(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_37(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_38(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_39(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁNortonBassModelǁscore__mutmut_40(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁNortonBassModelǁscore__mutmut_41(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)

        # y is expected to be of shape (n_samples, n_generations)
        # if y is 1D, reshape it
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        ss_res = B.sum((y - y_pred) ** 2)
        ss_tot = B.sum((y - B.mean(y, axis=0)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁNortonBassModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁscore__mutmut_1': xǁNortonBassModelǁscore__mutmut_1, 
        'xǁNortonBassModelǁscore__mutmut_2': xǁNortonBassModelǁscore__mutmut_2, 
        'xǁNortonBassModelǁscore__mutmut_3': xǁNortonBassModelǁscore__mutmut_3, 
        'xǁNortonBassModelǁscore__mutmut_4': xǁNortonBassModelǁscore__mutmut_4, 
        'xǁNortonBassModelǁscore__mutmut_5': xǁNortonBassModelǁscore__mutmut_5, 
        'xǁNortonBassModelǁscore__mutmut_6': xǁNortonBassModelǁscore__mutmut_6, 
        'xǁNortonBassModelǁscore__mutmut_7': xǁNortonBassModelǁscore__mutmut_7, 
        'xǁNortonBassModelǁscore__mutmut_8': xǁNortonBassModelǁscore__mutmut_8, 
        'xǁNortonBassModelǁscore__mutmut_9': xǁNortonBassModelǁscore__mutmut_9, 
        'xǁNortonBassModelǁscore__mutmut_10': xǁNortonBassModelǁscore__mutmut_10, 
        'xǁNortonBassModelǁscore__mutmut_11': xǁNortonBassModelǁscore__mutmut_11, 
        'xǁNortonBassModelǁscore__mutmut_12': xǁNortonBassModelǁscore__mutmut_12, 
        'xǁNortonBassModelǁscore__mutmut_13': xǁNortonBassModelǁscore__mutmut_13, 
        'xǁNortonBassModelǁscore__mutmut_14': xǁNortonBassModelǁscore__mutmut_14, 
        'xǁNortonBassModelǁscore__mutmut_15': xǁNortonBassModelǁscore__mutmut_15, 
        'xǁNortonBassModelǁscore__mutmut_16': xǁNortonBassModelǁscore__mutmut_16, 
        'xǁNortonBassModelǁscore__mutmut_17': xǁNortonBassModelǁscore__mutmut_17, 
        'xǁNortonBassModelǁscore__mutmut_18': xǁNortonBassModelǁscore__mutmut_18, 
        'xǁNortonBassModelǁscore__mutmut_19': xǁNortonBassModelǁscore__mutmut_19, 
        'xǁNortonBassModelǁscore__mutmut_20': xǁNortonBassModelǁscore__mutmut_20, 
        'xǁNortonBassModelǁscore__mutmut_21': xǁNortonBassModelǁscore__mutmut_21, 
        'xǁNortonBassModelǁscore__mutmut_22': xǁNortonBassModelǁscore__mutmut_22, 
        'xǁNortonBassModelǁscore__mutmut_23': xǁNortonBassModelǁscore__mutmut_23, 
        'xǁNortonBassModelǁscore__mutmut_24': xǁNortonBassModelǁscore__mutmut_24, 
        'xǁNortonBassModelǁscore__mutmut_25': xǁNortonBassModelǁscore__mutmut_25, 
        'xǁNortonBassModelǁscore__mutmut_26': xǁNortonBassModelǁscore__mutmut_26, 
        'xǁNortonBassModelǁscore__mutmut_27': xǁNortonBassModelǁscore__mutmut_27, 
        'xǁNortonBassModelǁscore__mutmut_28': xǁNortonBassModelǁscore__mutmut_28, 
        'xǁNortonBassModelǁscore__mutmut_29': xǁNortonBassModelǁscore__mutmut_29, 
        'xǁNortonBassModelǁscore__mutmut_30': xǁNortonBassModelǁscore__mutmut_30, 
        'xǁNortonBassModelǁscore__mutmut_31': xǁNortonBassModelǁscore__mutmut_31, 
        'xǁNortonBassModelǁscore__mutmut_32': xǁNortonBassModelǁscore__mutmut_32, 
        'xǁNortonBassModelǁscore__mutmut_33': xǁNortonBassModelǁscore__mutmut_33, 
        'xǁNortonBassModelǁscore__mutmut_34': xǁNortonBassModelǁscore__mutmut_34, 
        'xǁNortonBassModelǁscore__mutmut_35': xǁNortonBassModelǁscore__mutmut_35, 
        'xǁNortonBassModelǁscore__mutmut_36': xǁNortonBassModelǁscore__mutmut_36, 
        'xǁNortonBassModelǁscore__mutmut_37': xǁNortonBassModelǁscore__mutmut_37, 
        'xǁNortonBassModelǁscore__mutmut_38': xǁNortonBassModelǁscore__mutmut_38, 
        'xǁNortonBassModelǁscore__mutmut_39': xǁNortonBassModelǁscore__mutmut_39, 
        'xǁNortonBassModelǁscore__mutmut_40': xǁNortonBassModelǁscore__mutmut_40, 
        'xǁNortonBassModelǁscore__mutmut_41': xǁNortonBassModelǁscore__mutmut_41
    }
    xǁNortonBassModelǁscore__mutmut_orig.__name__ = 'xǁNortonBassModelǁscore'

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        args = [t, covariates]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁNortonBassModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁNortonBassModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_orig(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_1(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_2(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError(None)

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_3(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("XXModel has not been fitted yet. Call .fit() first.XX")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_4(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("model has not been fitted yet. call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_5(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("MODEL HAS NOT BEEN FITTED YET. CALL .FIT() FIRST.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_6(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = None
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_7(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(None, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_8(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, None)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_9(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_10(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, )
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_11(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = None

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_12(
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

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_13(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            None,
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_14(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(None, yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_15(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, None, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_16(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, None, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_17(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, None, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_18(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, None) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_19(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(yi, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_20(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, params, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_21(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, covariates, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_22(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, t) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_23(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, ) for ti, yi in zip(t, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_24(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(None, y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_25(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, None)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_26(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(y_pred)],
        )
        return rates

    def xǁNortonBassModelǁpredict_adoption_rate__mutmut_27(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = self.predict(t, covariates)
        params = [self._params[name] for name in self.param_names]

        rates = B.array(
            [self.differential_equation(ti, yi, params, covariates, t) for ti, yi in zip(t, )],
        )
        return rates
    
    xǁNortonBassModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁNortonBassModelǁpredict_adoption_rate__mutmut_1': xǁNortonBassModelǁpredict_adoption_rate__mutmut_1, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_2': xǁNortonBassModelǁpredict_adoption_rate__mutmut_2, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_3': xǁNortonBassModelǁpredict_adoption_rate__mutmut_3, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_4': xǁNortonBassModelǁpredict_adoption_rate__mutmut_4, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_5': xǁNortonBassModelǁpredict_adoption_rate__mutmut_5, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_6': xǁNortonBassModelǁpredict_adoption_rate__mutmut_6, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_7': xǁNortonBassModelǁpredict_adoption_rate__mutmut_7, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_8': xǁNortonBassModelǁpredict_adoption_rate__mutmut_8, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_9': xǁNortonBassModelǁpredict_adoption_rate__mutmut_9, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_10': xǁNortonBassModelǁpredict_adoption_rate__mutmut_10, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_11': xǁNortonBassModelǁpredict_adoption_rate__mutmut_11, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_12': xǁNortonBassModelǁpredict_adoption_rate__mutmut_12, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_13': xǁNortonBassModelǁpredict_adoption_rate__mutmut_13, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_14': xǁNortonBassModelǁpredict_adoption_rate__mutmut_14, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_15': xǁNortonBassModelǁpredict_adoption_rate__mutmut_15, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_16': xǁNortonBassModelǁpredict_adoption_rate__mutmut_16, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_17': xǁNortonBassModelǁpredict_adoption_rate__mutmut_17, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_18': xǁNortonBassModelǁpredict_adoption_rate__mutmut_18, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_19': xǁNortonBassModelǁpredict_adoption_rate__mutmut_19, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_20': xǁNortonBassModelǁpredict_adoption_rate__mutmut_20, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_21': xǁNortonBassModelǁpredict_adoption_rate__mutmut_21, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_22': xǁNortonBassModelǁpredict_adoption_rate__mutmut_22, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_23': xǁNortonBassModelǁpredict_adoption_rate__mutmut_23, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_24': xǁNortonBassModelǁpredict_adoption_rate__mutmut_24, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_25': xǁNortonBassModelǁpredict_adoption_rate__mutmut_25, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_26': xǁNortonBassModelǁpredict_adoption_rate__mutmut_26, 
        'xǁNortonBassModelǁpredict_adoption_rate__mutmut_27': xǁNortonBassModelǁpredict_adoption_rate__mutmut_27
    }
    xǁNortonBassModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁNortonBassModelǁpredict_adoption_rate'
