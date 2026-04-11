from collections.abc import Sequence

import numpy as np
import pandas as pd

from innovate.backend import current_backend as B
from innovate.base.base import DiffusionModel, Self
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


class MultiProductDiffusionModel(DiffusionModel):
    """Generic framework for multi-product/policy diffusion with competition and substitution."""

    def __init__(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        args = [p, Q, m, names]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁ__init____mutmut_orig'), object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁ__init____mutmut_mutants'), args, kwargs, self)

    def xǁMultiProductDiffusionModelǁ__init____mutmut_orig(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_1(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = None
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_2(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(None)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_3(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = None
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_4(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(None)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_5(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = None
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_6(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(None)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_7(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = None
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_8(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = None

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_9(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names and [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_10(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i - 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_11(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 2}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_12(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(None)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_13(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_14(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) or len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_15(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N or self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_16(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) != self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_17(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape != (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_18(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) != self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_19(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError(None)
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_20(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("XXDimensions of p, Q, and m must be consistent.XX")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_21(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("dimensions of p, q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_22(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("DIMENSIONS OF P, Q, AND M MUST BE CONSISTENT.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_23(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names or len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_24(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) == self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_25(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError(None)

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_26(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("XXLength of names must match the number of products (N).XX")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_27(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("length of names must match the number of products (n).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_28(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("LENGTH OF NAMES MUST MATCH THE NUMBER OF PRODUCTS (N).")

        self._params: dict[str, float] = {}

    def xǁMultiProductDiffusionModelǁ__init____mutmut_29(
        self,
        p: Sequence[float],  # length N: intrinsic adoption rates
        Q: Sequence[Sequence[float]],  # N x N matrix: interaction matrix (within- and cross-imitation)
        m: Sequence[float],  # length N: ultimate market potentials
        names: Sequence[str] | None = None,
    ):
        self.p = B.array(p)
        self.Q = B.array(Q)
        self.m = B.array(m)
        self.N = len(p)
        self.names = names or [f"Prod{i + 1}" for i in range(self.N)]

        if not (len(self.p) == self.N and self.Q.shape == (self.N, self.N) and len(self.m) == self.N):
            raise ValueError("Dimensions of p, Q, and m must be consistent.")
        if names and len(names) != self.N:
            raise ValueError("Length of names must match the number of products (N).")

        self._params: dict[str, float] = None
    
    xǁMultiProductDiffusionModelǁ__init____mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMultiProductDiffusionModelǁ__init____mutmut_1': xǁMultiProductDiffusionModelǁ__init____mutmut_1, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_2': xǁMultiProductDiffusionModelǁ__init____mutmut_2, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_3': xǁMultiProductDiffusionModelǁ__init____mutmut_3, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_4': xǁMultiProductDiffusionModelǁ__init____mutmut_4, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_5': xǁMultiProductDiffusionModelǁ__init____mutmut_5, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_6': xǁMultiProductDiffusionModelǁ__init____mutmut_6, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_7': xǁMultiProductDiffusionModelǁ__init____mutmut_7, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_8': xǁMultiProductDiffusionModelǁ__init____mutmut_8, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_9': xǁMultiProductDiffusionModelǁ__init____mutmut_9, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_10': xǁMultiProductDiffusionModelǁ__init____mutmut_10, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_11': xǁMultiProductDiffusionModelǁ__init____mutmut_11, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_12': xǁMultiProductDiffusionModelǁ__init____mutmut_12, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_13': xǁMultiProductDiffusionModelǁ__init____mutmut_13, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_14': xǁMultiProductDiffusionModelǁ__init____mutmut_14, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_15': xǁMultiProductDiffusionModelǁ__init____mutmut_15, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_16': xǁMultiProductDiffusionModelǁ__init____mutmut_16, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_17': xǁMultiProductDiffusionModelǁ__init____mutmut_17, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_18': xǁMultiProductDiffusionModelǁ__init____mutmut_18, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_19': xǁMultiProductDiffusionModelǁ__init____mutmut_19, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_20': xǁMultiProductDiffusionModelǁ__init____mutmut_20, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_21': xǁMultiProductDiffusionModelǁ__init____mutmut_21, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_22': xǁMultiProductDiffusionModelǁ__init____mutmut_22, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_23': xǁMultiProductDiffusionModelǁ__init____mutmut_23, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_24': xǁMultiProductDiffusionModelǁ__init____mutmut_24, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_25': xǁMultiProductDiffusionModelǁ__init____mutmut_25, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_26': xǁMultiProductDiffusionModelǁ__init____mutmut_26, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_27': xǁMultiProductDiffusionModelǁ__init____mutmut_27, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_28': xǁMultiProductDiffusionModelǁ__init____mutmut_28, 
        'xǁMultiProductDiffusionModelǁ__init____mutmut_29': xǁMultiProductDiffusionModelǁ__init____mutmut_29
    }
    xǁMultiProductDiffusionModelǁ__init____mutmut_orig.__name__ = 'xǁMultiProductDiffusionModelǁ__init__'

    def _rhs(self, y: Sequence[float], t: float) -> Sequence[float]:
        args = [y, t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁ_rhs__mutmut_orig'), object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁ_rhs__mutmut_mutants'), args, kwargs, self)

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_orig(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_1(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = None

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_2(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(None)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_3(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = None
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_4(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            None,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_5(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            None,
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_6(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            None,
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_7(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_8(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_9(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_10(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() == 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_11(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 1,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_12(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr * self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_13(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(None),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_14(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = None  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_15(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            None,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_16(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            None,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_17(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            None,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_18(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_19(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_20(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_21(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share >= 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_22(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 2.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_23(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            2.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_24(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = None  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_25(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(None, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_26(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, None)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_27(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_28(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, )  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_29(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = None  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_30(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p - imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_31(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = None

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_32(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(None, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_33(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, None, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_34(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, None)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_35(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_36(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_37(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, )

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_38(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m + y_arr < 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_39(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr <= 0, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_40(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 1, 0, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_41(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 1, self.m - y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_42(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m + y_arr)

        return force * remaining_potential

    def xǁMultiProductDiffusionModelǁ_rhs__mutmut_43(self, y: Sequence[float], t: float) -> Sequence[float]:
        """The right-hand side of the ODE system for N products."""
        # y: current cumulative adoptions for all N products
        # dNi = ( pi + sum_j Q[i,j] * (y[j]/m[j]) ) * (m[i] - y[i])

        # Ensure y is a numpy array for element-wise operations
        y_arr = B.array(y)

        # Avoid division by zero if m_j is zero, though m should be positive
        # Handle cases where y_j might exceed m_j slightly due to numerical issues
        adoption_share = B.where(
            self.m.flatten() != 0,
            y_arr / self.m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(
            adoption_share > 1.0,
            1.0,
            adoption_share,
        )  # Cap at 1.0

        imitation = B.matmul(self.Q, adoption_share)  # shape (N,)
        force = self.p + imitation  # shape (N,)

        # Ensure (m_i - y_i) does not go negative, which can happen with numerical solvers
        remaining_potential = B.where(self.m - y_arr < 0, 0, self.m - y_arr)

        return force / remaining_potential
    
    xǁMultiProductDiffusionModelǁ_rhs__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMultiProductDiffusionModelǁ_rhs__mutmut_1': xǁMultiProductDiffusionModelǁ_rhs__mutmut_1, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_2': xǁMultiProductDiffusionModelǁ_rhs__mutmut_2, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_3': xǁMultiProductDiffusionModelǁ_rhs__mutmut_3, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_4': xǁMultiProductDiffusionModelǁ_rhs__mutmut_4, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_5': xǁMultiProductDiffusionModelǁ_rhs__mutmut_5, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_6': xǁMultiProductDiffusionModelǁ_rhs__mutmut_6, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_7': xǁMultiProductDiffusionModelǁ_rhs__mutmut_7, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_8': xǁMultiProductDiffusionModelǁ_rhs__mutmut_8, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_9': xǁMultiProductDiffusionModelǁ_rhs__mutmut_9, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_10': xǁMultiProductDiffusionModelǁ_rhs__mutmut_10, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_11': xǁMultiProductDiffusionModelǁ_rhs__mutmut_11, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_12': xǁMultiProductDiffusionModelǁ_rhs__mutmut_12, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_13': xǁMultiProductDiffusionModelǁ_rhs__mutmut_13, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_14': xǁMultiProductDiffusionModelǁ_rhs__mutmut_14, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_15': xǁMultiProductDiffusionModelǁ_rhs__mutmut_15, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_16': xǁMultiProductDiffusionModelǁ_rhs__mutmut_16, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_17': xǁMultiProductDiffusionModelǁ_rhs__mutmut_17, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_18': xǁMultiProductDiffusionModelǁ_rhs__mutmut_18, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_19': xǁMultiProductDiffusionModelǁ_rhs__mutmut_19, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_20': xǁMultiProductDiffusionModelǁ_rhs__mutmut_20, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_21': xǁMultiProductDiffusionModelǁ_rhs__mutmut_21, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_22': xǁMultiProductDiffusionModelǁ_rhs__mutmut_22, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_23': xǁMultiProductDiffusionModelǁ_rhs__mutmut_23, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_24': xǁMultiProductDiffusionModelǁ_rhs__mutmut_24, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_25': xǁMultiProductDiffusionModelǁ_rhs__mutmut_25, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_26': xǁMultiProductDiffusionModelǁ_rhs__mutmut_26, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_27': xǁMultiProductDiffusionModelǁ_rhs__mutmut_27, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_28': xǁMultiProductDiffusionModelǁ_rhs__mutmut_28, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_29': xǁMultiProductDiffusionModelǁ_rhs__mutmut_29, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_30': xǁMultiProductDiffusionModelǁ_rhs__mutmut_30, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_31': xǁMultiProductDiffusionModelǁ_rhs__mutmut_31, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_32': xǁMultiProductDiffusionModelǁ_rhs__mutmut_32, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_33': xǁMultiProductDiffusionModelǁ_rhs__mutmut_33, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_34': xǁMultiProductDiffusionModelǁ_rhs__mutmut_34, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_35': xǁMultiProductDiffusionModelǁ_rhs__mutmut_35, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_36': xǁMultiProductDiffusionModelǁ_rhs__mutmut_36, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_37': xǁMultiProductDiffusionModelǁ_rhs__mutmut_37, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_38': xǁMultiProductDiffusionModelǁ_rhs__mutmut_38, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_39': xǁMultiProductDiffusionModelǁ_rhs__mutmut_39, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_40': xǁMultiProductDiffusionModelǁ_rhs__mutmut_40, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_41': xǁMultiProductDiffusionModelǁ_rhs__mutmut_41, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_42': xǁMultiProductDiffusionModelǁ_rhs__mutmut_42, 
        'xǁMultiProductDiffusionModelǁ_rhs__mutmut_43': xǁMultiProductDiffusionModelǁ_rhs__mutmut_43
    }
    xǁMultiProductDiffusionModelǁ_rhs__mutmut_orig.__name__ = 'xǁMultiProductDiffusionModelǁ_rhs'

    def predict(self, t: Sequence[float]) -> pd.DataFrame:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁpredict__mutmut_orig'), object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁpredict__mutmut_mutants'), args, kwargs, self)

    def xǁMultiProductDiffusionModelǁpredict__mutmut_orig(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_1(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ or (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_2(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_3(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None and self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_4(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None and self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_5(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is not None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_6(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is not None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_7(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is not None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_8(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                None,
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_9(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "XXModel parameters are not set. Call .fit() or initialize with p, Q, m.XX",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_10(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "model parameters are not set. call .fit() or initialize with p, q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_11(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "MODEL PARAMETERS ARE NOT SET. CALL .FIT() OR INITIALIZE WITH P, Q, M.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_12(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = None
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_13(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(None)
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_14(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get(None, self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_15(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", None))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_16(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get(self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_17(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", ))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_18(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("XXpXX", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_19(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("P", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_20(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = None
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_21(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(None)
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_22(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get(None, self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_23(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", None))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_24(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get(self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_25(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", ))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_26(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("XXQXX", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_27(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_28(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = None

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_29(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(None)

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_30(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get(None, self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_31(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", None))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_32(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get(self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_33(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", ))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_34(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("XXmXX", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_35(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("M", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_36(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = None

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_37(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros(None)

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_38(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = None

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_39(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(None, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_40(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, None)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_41(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_42(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, )

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_43(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = None

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_44(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(None, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_45(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, None, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_46(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, None)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_47(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_48(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_49(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, )

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_50(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = None
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_51(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(None, index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_52(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=None, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_53(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, columns=None)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_54(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(index=t, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_55(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, columns=self.names)
        return df

    def xǁMultiProductDiffusionModelǁpredict__mutmut_56(self, t: Sequence[float]) -> pd.DataFrame:
        # Ensure parameters are set (either by init or by a fitter)
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model parameters are not set. Call .fit() or initialize with p, Q, m.",
            )

        # If fit was called, use the stored parameters. Otherwise, use initial ones.
        current_p = B.array(self.params_.get("p", self.p))
        current_Q = B.array(self.params_.get("Q", self.Q))
        current_m = B.array(self.params_.get("m", self.m))

        # Initial conditions: start with 0 adoptions for all products
        y0 = B.zeros((self.N,))

        # Solve the ODE system
        # The _rhs function expects (y, t) for scipy.integrate.odeint
        # We need to pass the current parameters (p, Q, m) to the _rhs function
        # This requires a partial function or passing them as args to solve_ode

        # For scipy.integrate.odeint, the signature is func(y, t, ...args)
        # So, we need to pass p, Q, m as args to solve_ode

        # Temporarily store current parameters for _rhs access if needed by odeint
        # This is a common pattern when using odeint with class methods
        self._current_ode_params = (current_p, current_Q, current_m)

        def ode_func(y, t_val):
            # This wrapper allows _rhs to access self.p, self.Q, self.m
            # and matches the (y, t) signature expected by odeint
            # Note: self._rhs expects (y, t) as per the backend protocol
            return self._rhs(y, t_val)

        sol = B.solve_ode(ode_func, y0, t)

        # Convert solution to pandas DataFrame
        df = pd.DataFrame(sol, index=t, )
        return df
    
    xǁMultiProductDiffusionModelǁpredict__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMultiProductDiffusionModelǁpredict__mutmut_1': xǁMultiProductDiffusionModelǁpredict__mutmut_1, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_2': xǁMultiProductDiffusionModelǁpredict__mutmut_2, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_3': xǁMultiProductDiffusionModelǁpredict__mutmut_3, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_4': xǁMultiProductDiffusionModelǁpredict__mutmut_4, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_5': xǁMultiProductDiffusionModelǁpredict__mutmut_5, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_6': xǁMultiProductDiffusionModelǁpredict__mutmut_6, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_7': xǁMultiProductDiffusionModelǁpredict__mutmut_7, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_8': xǁMultiProductDiffusionModelǁpredict__mutmut_8, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_9': xǁMultiProductDiffusionModelǁpredict__mutmut_9, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_10': xǁMultiProductDiffusionModelǁpredict__mutmut_10, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_11': xǁMultiProductDiffusionModelǁpredict__mutmut_11, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_12': xǁMultiProductDiffusionModelǁpredict__mutmut_12, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_13': xǁMultiProductDiffusionModelǁpredict__mutmut_13, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_14': xǁMultiProductDiffusionModelǁpredict__mutmut_14, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_15': xǁMultiProductDiffusionModelǁpredict__mutmut_15, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_16': xǁMultiProductDiffusionModelǁpredict__mutmut_16, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_17': xǁMultiProductDiffusionModelǁpredict__mutmut_17, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_18': xǁMultiProductDiffusionModelǁpredict__mutmut_18, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_19': xǁMultiProductDiffusionModelǁpredict__mutmut_19, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_20': xǁMultiProductDiffusionModelǁpredict__mutmut_20, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_21': xǁMultiProductDiffusionModelǁpredict__mutmut_21, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_22': xǁMultiProductDiffusionModelǁpredict__mutmut_22, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_23': xǁMultiProductDiffusionModelǁpredict__mutmut_23, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_24': xǁMultiProductDiffusionModelǁpredict__mutmut_24, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_25': xǁMultiProductDiffusionModelǁpredict__mutmut_25, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_26': xǁMultiProductDiffusionModelǁpredict__mutmut_26, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_27': xǁMultiProductDiffusionModelǁpredict__mutmut_27, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_28': xǁMultiProductDiffusionModelǁpredict__mutmut_28, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_29': xǁMultiProductDiffusionModelǁpredict__mutmut_29, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_30': xǁMultiProductDiffusionModelǁpredict__mutmut_30, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_31': xǁMultiProductDiffusionModelǁpredict__mutmut_31, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_32': xǁMultiProductDiffusionModelǁpredict__mutmut_32, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_33': xǁMultiProductDiffusionModelǁpredict__mutmut_33, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_34': xǁMultiProductDiffusionModelǁpredict__mutmut_34, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_35': xǁMultiProductDiffusionModelǁpredict__mutmut_35, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_36': xǁMultiProductDiffusionModelǁpredict__mutmut_36, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_37': xǁMultiProductDiffusionModelǁpredict__mutmut_37, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_38': xǁMultiProductDiffusionModelǁpredict__mutmut_38, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_39': xǁMultiProductDiffusionModelǁpredict__mutmut_39, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_40': xǁMultiProductDiffusionModelǁpredict__mutmut_40, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_41': xǁMultiProductDiffusionModelǁpredict__mutmut_41, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_42': xǁMultiProductDiffusionModelǁpredict__mutmut_42, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_43': xǁMultiProductDiffusionModelǁpredict__mutmut_43, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_44': xǁMultiProductDiffusionModelǁpredict__mutmut_44, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_45': xǁMultiProductDiffusionModelǁpredict__mutmut_45, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_46': xǁMultiProductDiffusionModelǁpredict__mutmut_46, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_47': xǁMultiProductDiffusionModelǁpredict__mutmut_47, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_48': xǁMultiProductDiffusionModelǁpredict__mutmut_48, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_49': xǁMultiProductDiffusionModelǁpredict__mutmut_49, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_50': xǁMultiProductDiffusionModelǁpredict__mutmut_50, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_51': xǁMultiProductDiffusionModelǁpredict__mutmut_51, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_52': xǁMultiProductDiffusionModelǁpredict__mutmut_52, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_53': xǁMultiProductDiffusionModelǁpredict__mutmut_53, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_54': xǁMultiProductDiffusionModelǁpredict__mutmut_54, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_55': xǁMultiProductDiffusionModelǁpredict__mutmut_55, 
        'xǁMultiProductDiffusionModelǁpredict__mutmut_56': xǁMultiProductDiffusionModelǁpredict__mutmut_56
    }
    xǁMultiProductDiffusionModelǁpredict__mutmut_orig.__name__ = 'xǁMultiProductDiffusionModelǁpredict'

    @staticmethod
    def differential_equation(y, t, params):
        """Differential equations for the multi-product model."""
        p, Q, m = params
        y_arr = B.array(y)
        adoption_share = B.where(
            m.flatten() != 0,
            y_arr / m.flatten(),
            B.zeros_like(y_arr),
        )
        adoption_share = B.where(adoption_share > 1.0, 1.0, adoption_share)
        imitation = B.matmul(Q, adoption_share)
        force = p + imitation
        remaining_potential = B.where(m - y_arr < 0, 0, m - y_arr)
        return force * remaining_potential

    def fit(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        args = [t, y]# type: ignore
        kwargs = {**kwargs}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁfit__mutmut_orig'), object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁfit__mutmut_mutants'), args, kwargs, self)

    def xǁMultiProductDiffusionModelǁfit__mutmut_orig(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_1(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = None
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_2(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(None)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_3(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 and y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_4(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim == 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_5(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 3 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_6(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[2] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_7(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] == self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_8(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError(None)

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_9(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("XXObserved data must be a 2D array with N columnsXX")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_10(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("observed data must be a 2d array with n columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_11(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("OBSERVED DATA MUST BE A 2D ARRAY WITH N COLUMNS")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_12(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = None

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_13(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(None)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_14(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate(None)

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_15(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = None
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_16(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = None
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_17(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end - self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_18(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N / self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_19(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = None
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_20(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(None)
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_21(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = None
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_22(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(None, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_23(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, None)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_24(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_25(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, )
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_26(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(None).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_27(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = None
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_28(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(None)
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_29(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end - self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_30(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = None
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_31(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(None, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_32(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, None)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_33(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_34(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, )
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_35(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = None
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_36(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(None)
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_37(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get(None, self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_38(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", None))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_39(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get(self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_40(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", ))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_41(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("XXpXX", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_42(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("P", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_43(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = None
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_44(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(None)
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_45(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get(None, self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_46(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", None))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_47(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get(self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_48(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", ))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_49(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("XXQXX", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_50(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_51(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = None
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_52(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(None)
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_53(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get(None, self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_54(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", None))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_55(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get(self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_56(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", ))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_57(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("XXmXX", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_58(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("M", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_59(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = None

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_60(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(None, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_61(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, None, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_62(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, None)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_63(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_64(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_65(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, )

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_66(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = None

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_67(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(None, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_68(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, None)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_69(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_70(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, )

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_71(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=1.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_72(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] / size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_73(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = None
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_74(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get(None, _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_75(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", None)
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_76(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get(_default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_77(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", )
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_78(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("XXpXX", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_79(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("P", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_80(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(None))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_81(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = None
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_82(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get(None, [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_83(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", None)
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_84(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get([(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_85(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", )
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_86(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("XXQXX", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_87(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_88(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] / (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_89(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N / self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_90(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = None
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_91(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get(None, _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_92(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", None)
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_93(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get(_default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_94(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", )
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_95(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("XXmXX", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_96(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("M", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_97(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(None))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_98(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = None

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_99(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q - b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_100(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p - b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_101(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = None
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_102(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(None)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_103(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = None
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_104(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(None)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_105(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = None
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_106(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(None)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_107(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = None
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_108(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(None)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_109(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = None
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_110(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(None).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_111(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum(None)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_112(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) * 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_113(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr + pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_114(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 3)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_115(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = None

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_116(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(None, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_117(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, None, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_118(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=None, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_119(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method=None, **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_120(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_121(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_122(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_123(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_124(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", )

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_125(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="XXL-BFGS-BXX", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_126(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="l-bfgs-b", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_127(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_128(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(None)

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_129(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = None
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_130(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(None)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_131(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = None
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_132(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(None)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_133(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = None
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_134(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(None)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_135(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = None
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_136(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(None)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_137(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = None
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_138(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"XXpXX": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_139(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"P": opt_p.tolist(), "Q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_140(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "XXQXX": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_141(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "q": opt_Q.tolist(), "m": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_142(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "XXmXX": opt_m.tolist()}
        return self

    def xǁMultiProductDiffusionModelǁfit__mutmut_143(self, t: Sequence[float], y: Sequence[Sequence[float]], **kwargs) -> Self:
        """Fit model parameters by minimizing squared prediction error."""
        from scipy.optimize import minimize

        y_arr = np.array(y)
        if y_arr.ndim != 2 or y_arr.shape[1] != self.N:
            raise ValueError("Observed data must be a 2D array with N columns")

        t_arr = np.array(t)

        def flatten(p_vec, Q_mat, m_vec):
            return np.concatenate([p_vec, Q_mat.flatten(), m_vec])

        def unflatten(params):
            p_end = self.N
            Q_end = p_end + self.N * self.N
            p_vec = np.array(params[:p_end])
            Q_mat = np.array(params[p_end:Q_end]).reshape(self.N, self.N)
            m_vec = np.array(params[Q_end : Q_end + self.N])
            return p_vec, Q_mat, m_vec

        guesses = self.initial_guesses(t_arr, y_arr)
        p0 = np.array(guesses.get("p", self.p))
        Q0 = np.array(guesses.get("Q", self.Q))
        m0 = np.array(guesses.get("m", self.m))
        x0 = flatten(p0, Q0, m0)

        bounds_dict = self.bounds(t_arr, y_arr)

        def _default_bounds(size, lb=0.0):
            return [(lb, None)] * size

        b_p = bounds_dict.get("p", _default_bounds(self.N))
        b_Q = bounds_dict.get("Q", [(None, None)] * (self.N * self.N))
        b_m = bounds_dict.get("m", _default_bounds(self.N))
        bounds = b_p + b_Q + b_m

        def objective(params):
            p_vec, Q_mat, m_vec = unflatten(params)
            self.p = B.array(p_vec)
            self.Q = B.array(Q_mat)
            self.m = B.array(m_vec)
            pred = self.predict(t_arr).values
            return np.sum((y_arr - pred) ** 2)

        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B", **kwargs)

        if not result.success:
            raise RuntimeError(f"Fitting failed: {result.message}")

        opt_p, opt_Q, opt_m = unflatten(result.x)
        self.p = B.array(opt_p)
        self.Q = B.array(opt_Q)
        self.m = B.array(opt_m)
        self._params = {"p": opt_p.tolist(), "Q": opt_Q.tolist(), "M": opt_m.tolist()}
        return self
    
    xǁMultiProductDiffusionModelǁfit__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMultiProductDiffusionModelǁfit__mutmut_1': xǁMultiProductDiffusionModelǁfit__mutmut_1, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_2': xǁMultiProductDiffusionModelǁfit__mutmut_2, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_3': xǁMultiProductDiffusionModelǁfit__mutmut_3, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_4': xǁMultiProductDiffusionModelǁfit__mutmut_4, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_5': xǁMultiProductDiffusionModelǁfit__mutmut_5, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_6': xǁMultiProductDiffusionModelǁfit__mutmut_6, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_7': xǁMultiProductDiffusionModelǁfit__mutmut_7, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_8': xǁMultiProductDiffusionModelǁfit__mutmut_8, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_9': xǁMultiProductDiffusionModelǁfit__mutmut_9, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_10': xǁMultiProductDiffusionModelǁfit__mutmut_10, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_11': xǁMultiProductDiffusionModelǁfit__mutmut_11, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_12': xǁMultiProductDiffusionModelǁfit__mutmut_12, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_13': xǁMultiProductDiffusionModelǁfit__mutmut_13, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_14': xǁMultiProductDiffusionModelǁfit__mutmut_14, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_15': xǁMultiProductDiffusionModelǁfit__mutmut_15, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_16': xǁMultiProductDiffusionModelǁfit__mutmut_16, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_17': xǁMultiProductDiffusionModelǁfit__mutmut_17, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_18': xǁMultiProductDiffusionModelǁfit__mutmut_18, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_19': xǁMultiProductDiffusionModelǁfit__mutmut_19, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_20': xǁMultiProductDiffusionModelǁfit__mutmut_20, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_21': xǁMultiProductDiffusionModelǁfit__mutmut_21, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_22': xǁMultiProductDiffusionModelǁfit__mutmut_22, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_23': xǁMultiProductDiffusionModelǁfit__mutmut_23, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_24': xǁMultiProductDiffusionModelǁfit__mutmut_24, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_25': xǁMultiProductDiffusionModelǁfit__mutmut_25, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_26': xǁMultiProductDiffusionModelǁfit__mutmut_26, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_27': xǁMultiProductDiffusionModelǁfit__mutmut_27, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_28': xǁMultiProductDiffusionModelǁfit__mutmut_28, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_29': xǁMultiProductDiffusionModelǁfit__mutmut_29, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_30': xǁMultiProductDiffusionModelǁfit__mutmut_30, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_31': xǁMultiProductDiffusionModelǁfit__mutmut_31, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_32': xǁMultiProductDiffusionModelǁfit__mutmut_32, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_33': xǁMultiProductDiffusionModelǁfit__mutmut_33, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_34': xǁMultiProductDiffusionModelǁfit__mutmut_34, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_35': xǁMultiProductDiffusionModelǁfit__mutmut_35, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_36': xǁMultiProductDiffusionModelǁfit__mutmut_36, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_37': xǁMultiProductDiffusionModelǁfit__mutmut_37, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_38': xǁMultiProductDiffusionModelǁfit__mutmut_38, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_39': xǁMultiProductDiffusionModelǁfit__mutmut_39, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_40': xǁMultiProductDiffusionModelǁfit__mutmut_40, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_41': xǁMultiProductDiffusionModelǁfit__mutmut_41, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_42': xǁMultiProductDiffusionModelǁfit__mutmut_42, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_43': xǁMultiProductDiffusionModelǁfit__mutmut_43, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_44': xǁMultiProductDiffusionModelǁfit__mutmut_44, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_45': xǁMultiProductDiffusionModelǁfit__mutmut_45, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_46': xǁMultiProductDiffusionModelǁfit__mutmut_46, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_47': xǁMultiProductDiffusionModelǁfit__mutmut_47, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_48': xǁMultiProductDiffusionModelǁfit__mutmut_48, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_49': xǁMultiProductDiffusionModelǁfit__mutmut_49, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_50': xǁMultiProductDiffusionModelǁfit__mutmut_50, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_51': xǁMultiProductDiffusionModelǁfit__mutmut_51, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_52': xǁMultiProductDiffusionModelǁfit__mutmut_52, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_53': xǁMultiProductDiffusionModelǁfit__mutmut_53, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_54': xǁMultiProductDiffusionModelǁfit__mutmut_54, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_55': xǁMultiProductDiffusionModelǁfit__mutmut_55, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_56': xǁMultiProductDiffusionModelǁfit__mutmut_56, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_57': xǁMultiProductDiffusionModelǁfit__mutmut_57, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_58': xǁMultiProductDiffusionModelǁfit__mutmut_58, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_59': xǁMultiProductDiffusionModelǁfit__mutmut_59, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_60': xǁMultiProductDiffusionModelǁfit__mutmut_60, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_61': xǁMultiProductDiffusionModelǁfit__mutmut_61, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_62': xǁMultiProductDiffusionModelǁfit__mutmut_62, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_63': xǁMultiProductDiffusionModelǁfit__mutmut_63, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_64': xǁMultiProductDiffusionModelǁfit__mutmut_64, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_65': xǁMultiProductDiffusionModelǁfit__mutmut_65, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_66': xǁMultiProductDiffusionModelǁfit__mutmut_66, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_67': xǁMultiProductDiffusionModelǁfit__mutmut_67, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_68': xǁMultiProductDiffusionModelǁfit__mutmut_68, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_69': xǁMultiProductDiffusionModelǁfit__mutmut_69, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_70': xǁMultiProductDiffusionModelǁfit__mutmut_70, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_71': xǁMultiProductDiffusionModelǁfit__mutmut_71, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_72': xǁMultiProductDiffusionModelǁfit__mutmut_72, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_73': xǁMultiProductDiffusionModelǁfit__mutmut_73, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_74': xǁMultiProductDiffusionModelǁfit__mutmut_74, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_75': xǁMultiProductDiffusionModelǁfit__mutmut_75, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_76': xǁMultiProductDiffusionModelǁfit__mutmut_76, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_77': xǁMultiProductDiffusionModelǁfit__mutmut_77, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_78': xǁMultiProductDiffusionModelǁfit__mutmut_78, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_79': xǁMultiProductDiffusionModelǁfit__mutmut_79, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_80': xǁMultiProductDiffusionModelǁfit__mutmut_80, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_81': xǁMultiProductDiffusionModelǁfit__mutmut_81, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_82': xǁMultiProductDiffusionModelǁfit__mutmut_82, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_83': xǁMultiProductDiffusionModelǁfit__mutmut_83, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_84': xǁMultiProductDiffusionModelǁfit__mutmut_84, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_85': xǁMultiProductDiffusionModelǁfit__mutmut_85, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_86': xǁMultiProductDiffusionModelǁfit__mutmut_86, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_87': xǁMultiProductDiffusionModelǁfit__mutmut_87, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_88': xǁMultiProductDiffusionModelǁfit__mutmut_88, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_89': xǁMultiProductDiffusionModelǁfit__mutmut_89, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_90': xǁMultiProductDiffusionModelǁfit__mutmut_90, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_91': xǁMultiProductDiffusionModelǁfit__mutmut_91, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_92': xǁMultiProductDiffusionModelǁfit__mutmut_92, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_93': xǁMultiProductDiffusionModelǁfit__mutmut_93, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_94': xǁMultiProductDiffusionModelǁfit__mutmut_94, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_95': xǁMultiProductDiffusionModelǁfit__mutmut_95, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_96': xǁMultiProductDiffusionModelǁfit__mutmut_96, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_97': xǁMultiProductDiffusionModelǁfit__mutmut_97, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_98': xǁMultiProductDiffusionModelǁfit__mutmut_98, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_99': xǁMultiProductDiffusionModelǁfit__mutmut_99, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_100': xǁMultiProductDiffusionModelǁfit__mutmut_100, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_101': xǁMultiProductDiffusionModelǁfit__mutmut_101, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_102': xǁMultiProductDiffusionModelǁfit__mutmut_102, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_103': xǁMultiProductDiffusionModelǁfit__mutmut_103, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_104': xǁMultiProductDiffusionModelǁfit__mutmut_104, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_105': xǁMultiProductDiffusionModelǁfit__mutmut_105, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_106': xǁMultiProductDiffusionModelǁfit__mutmut_106, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_107': xǁMultiProductDiffusionModelǁfit__mutmut_107, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_108': xǁMultiProductDiffusionModelǁfit__mutmut_108, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_109': xǁMultiProductDiffusionModelǁfit__mutmut_109, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_110': xǁMultiProductDiffusionModelǁfit__mutmut_110, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_111': xǁMultiProductDiffusionModelǁfit__mutmut_111, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_112': xǁMultiProductDiffusionModelǁfit__mutmut_112, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_113': xǁMultiProductDiffusionModelǁfit__mutmut_113, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_114': xǁMultiProductDiffusionModelǁfit__mutmut_114, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_115': xǁMultiProductDiffusionModelǁfit__mutmut_115, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_116': xǁMultiProductDiffusionModelǁfit__mutmut_116, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_117': xǁMultiProductDiffusionModelǁfit__mutmut_117, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_118': xǁMultiProductDiffusionModelǁfit__mutmut_118, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_119': xǁMultiProductDiffusionModelǁfit__mutmut_119, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_120': xǁMultiProductDiffusionModelǁfit__mutmut_120, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_121': xǁMultiProductDiffusionModelǁfit__mutmut_121, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_122': xǁMultiProductDiffusionModelǁfit__mutmut_122, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_123': xǁMultiProductDiffusionModelǁfit__mutmut_123, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_124': xǁMultiProductDiffusionModelǁfit__mutmut_124, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_125': xǁMultiProductDiffusionModelǁfit__mutmut_125, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_126': xǁMultiProductDiffusionModelǁfit__mutmut_126, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_127': xǁMultiProductDiffusionModelǁfit__mutmut_127, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_128': xǁMultiProductDiffusionModelǁfit__mutmut_128, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_129': xǁMultiProductDiffusionModelǁfit__mutmut_129, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_130': xǁMultiProductDiffusionModelǁfit__mutmut_130, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_131': xǁMultiProductDiffusionModelǁfit__mutmut_131, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_132': xǁMultiProductDiffusionModelǁfit__mutmut_132, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_133': xǁMultiProductDiffusionModelǁfit__mutmut_133, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_134': xǁMultiProductDiffusionModelǁfit__mutmut_134, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_135': xǁMultiProductDiffusionModelǁfit__mutmut_135, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_136': xǁMultiProductDiffusionModelǁfit__mutmut_136, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_137': xǁMultiProductDiffusionModelǁfit__mutmut_137, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_138': xǁMultiProductDiffusionModelǁfit__mutmut_138, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_139': xǁMultiProductDiffusionModelǁfit__mutmut_139, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_140': xǁMultiProductDiffusionModelǁfit__mutmut_140, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_141': xǁMultiProductDiffusionModelǁfit__mutmut_141, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_142': xǁMultiProductDiffusionModelǁfit__mutmut_142, 
        'xǁMultiProductDiffusionModelǁfit__mutmut_143': xǁMultiProductDiffusionModelǁfit__mutmut_143
    }
    xǁMultiProductDiffusionModelǁfit__mutmut_orig.__name__ = 'xǁMultiProductDiffusionModelǁfit'

    def score(self, t: Sequence[float], y: pd.DataFrame) -> float:
        args = [t, y]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁscore__mutmut_orig'), object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁscore__mutmut_mutants'), args, kwargs, self)

    def xǁMultiProductDiffusionModelǁscore__mutmut_orig(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_1(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ or (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_2(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_3(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None and self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_4(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None and self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_5(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is not None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_6(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is not None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_7(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is not None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_8(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                None,
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_9(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "XXModel has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.XX",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_10(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "model has not been fitted or initialized with parameters yet. call .fit() or initialize with p, q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_11(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "MODEL HAS NOT BEEN FITTED OR INITIALIZED WITH PARAMETERS YET. CALL .FIT() OR INITIALIZE WITH P, Q, M.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_12(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = None

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_13(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(None)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_14(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_15(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(None):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_16(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name not in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_17(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                None,
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_18(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = None
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_19(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(None)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_20(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = None

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_21(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(None)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_22(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = None
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_23(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum(None)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_24(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) * 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_25(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) + B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_26(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(None) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_27(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(None)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_28(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 3)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_29(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = None
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_30(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum(None)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_31(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) * 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_32(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) + B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_33(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(None) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_34(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(None)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_35(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(None))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_36(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 3)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_37(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 + (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_38(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 2 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_39(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res * ss_tot) if ss_tot > 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_40(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot >= 0 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_41(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 1 else 0.0

    def xǁMultiProductDiffusionModelǁscore__mutmut_42(self, t: Sequence[float], y: pd.DataFrame) -> float:
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(
                "Model has not been fitted or initialized with parameters yet. Call .fit() or initialize with p, Q, m.",
            )

        y_pred_df = self.predict(t)

        # Ensure y contains all product names and is in the correct order
        if not all(name in y.columns for name in self.names):
            raise ValueError(
                f"Observed data DataFrame must contain columns for all products: {self.names}",
            )

        y_obs_aligned = y[list(self.names)].values.flatten()
        y_pred_aligned = y_pred_df[list(self.names)].values.flatten()

        ss_res = B.sum((B.array(y_obs_aligned) - B.array(y_pred_aligned)) ** 2)
        ss_tot = B.sum((B.array(y_obs_aligned) - B.mean(B.array(y_obs_aligned))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    xǁMultiProductDiffusionModelǁscore__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMultiProductDiffusionModelǁscore__mutmut_1': xǁMultiProductDiffusionModelǁscore__mutmut_1, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_2': xǁMultiProductDiffusionModelǁscore__mutmut_2, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_3': xǁMultiProductDiffusionModelǁscore__mutmut_3, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_4': xǁMultiProductDiffusionModelǁscore__mutmut_4, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_5': xǁMultiProductDiffusionModelǁscore__mutmut_5, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_6': xǁMultiProductDiffusionModelǁscore__mutmut_6, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_7': xǁMultiProductDiffusionModelǁscore__mutmut_7, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_8': xǁMultiProductDiffusionModelǁscore__mutmut_8, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_9': xǁMultiProductDiffusionModelǁscore__mutmut_9, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_10': xǁMultiProductDiffusionModelǁscore__mutmut_10, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_11': xǁMultiProductDiffusionModelǁscore__mutmut_11, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_12': xǁMultiProductDiffusionModelǁscore__mutmut_12, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_13': xǁMultiProductDiffusionModelǁscore__mutmut_13, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_14': xǁMultiProductDiffusionModelǁscore__mutmut_14, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_15': xǁMultiProductDiffusionModelǁscore__mutmut_15, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_16': xǁMultiProductDiffusionModelǁscore__mutmut_16, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_17': xǁMultiProductDiffusionModelǁscore__mutmut_17, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_18': xǁMultiProductDiffusionModelǁscore__mutmut_18, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_19': xǁMultiProductDiffusionModelǁscore__mutmut_19, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_20': xǁMultiProductDiffusionModelǁscore__mutmut_20, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_21': xǁMultiProductDiffusionModelǁscore__mutmut_21, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_22': xǁMultiProductDiffusionModelǁscore__mutmut_22, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_23': xǁMultiProductDiffusionModelǁscore__mutmut_23, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_24': xǁMultiProductDiffusionModelǁscore__mutmut_24, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_25': xǁMultiProductDiffusionModelǁscore__mutmut_25, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_26': xǁMultiProductDiffusionModelǁscore__mutmut_26, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_27': xǁMultiProductDiffusionModelǁscore__mutmut_27, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_28': xǁMultiProductDiffusionModelǁscore__mutmut_28, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_29': xǁMultiProductDiffusionModelǁscore__mutmut_29, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_30': xǁMultiProductDiffusionModelǁscore__mutmut_30, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_31': xǁMultiProductDiffusionModelǁscore__mutmut_31, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_32': xǁMultiProductDiffusionModelǁscore__mutmut_32, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_33': xǁMultiProductDiffusionModelǁscore__mutmut_33, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_34': xǁMultiProductDiffusionModelǁscore__mutmut_34, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_35': xǁMultiProductDiffusionModelǁscore__mutmut_35, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_36': xǁMultiProductDiffusionModelǁscore__mutmut_36, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_37': xǁMultiProductDiffusionModelǁscore__mutmut_37, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_38': xǁMultiProductDiffusionModelǁscore__mutmut_38, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_39': xǁMultiProductDiffusionModelǁscore__mutmut_39, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_40': xǁMultiProductDiffusionModelǁscore__mutmut_40, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_41': xǁMultiProductDiffusionModelǁscore__mutmut_41, 
        'xǁMultiProductDiffusionModelǁscore__mutmut_42': xǁMultiProductDiffusionModelǁscore__mutmut_42
    }
    xǁMultiProductDiffusionModelǁscore__mutmut_orig.__name__ = 'xǁMultiProductDiffusionModelǁscore'

    @property
    def params_(self) -> dict[str, float | list[float] | list[list[float]]]:
        # Return the parameters that were either initialized or fitted
        if self._params:
            return self._params
        return {"p": self.p.tolist(), "Q": self.Q.tolist(), "m": self.m.tolist()}

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float]) -> pd.DataFrame:
        args = [t]# type: ignore
        kwargs = {}# type: ignore
        return _mutmut_trampoline(object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_orig'), object.__getattribute__(self, 'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_mutants'), args, kwargs, self)

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_orig(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_1(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ or (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_2(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_3(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None and self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_4(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None and self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_5(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is not None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_6(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is not None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_7(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is not None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_8(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError(None)

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_9(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("XXModel parameters are not set. Call .fit() or initialize with p, Q, m.XX")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_10(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("model parameters are not set. call .fit() or initialize with p, q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_11(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("MODEL PARAMETERS ARE NOT SET. CALL .FIT() OR INITIALIZE WITH P, Q, M.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_12(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = None

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_13(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(None)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_14(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = None

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_15(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(None)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_16(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = None

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_17(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(None):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_18(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i != 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_19(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 1:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_20(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = None
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_21(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[1].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_22(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = None

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_23(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = None
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_24(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(None, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_25(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, None)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_26(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_27(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, )
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_28(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(None)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_29(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = None

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_30(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(None, index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_31(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=None, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_32(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, columns=None)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_33(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(index=t, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_34(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, columns=self.names)

        return rates_df

    def xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_35(self, t: Sequence[float]) -> pd.DataFrame:
        """
        Predict the rate of new adoptions per time period for each product.

        This method calculates the derivative of cumulative adoptions, representing
        the instantaneous adoption rate for each product in the multi-product system.

        Parameters
        ----------
        t : Sequence[float]
            Time points at which to predict adoption rates

        Returns
        -------
        pd.DataFrame
            DataFrame with adoption rates for each product at each time point.
            Columns correspond to product names, rows to time points.

        Raises
        ------
        RuntimeError
            If model parameters are not set (model not fitted or initialized)
        """
        if not self.params_ and (self.p is None or self.Q is None or self.m is None):
            raise RuntimeError("Model parameters are not set. Call .fit() or initialize with p, Q, m.")

        # Get cumulative predictions
        cumulative_df = self.predict(t)

        # Calculate adoption rates using numerical differentiation
        t_arr = B.array(t)

        # For the first point, use the differential equation directly
        adoption_rates = []

        for i, time_point in enumerate(t_arr):
            if i == 0:
                # For the first point, evaluate the differential equation at t=0
                y_current = cumulative_df.iloc[0].values
            else:
                y_current = cumulative_df.iloc[i].values

            # Use the differential equation to get instantaneous rates
            rate = self._rhs(y_current, time_point)
            adoption_rates.append(rate)

        # Convert to DataFrame with same structure as predict output
        rates_df = pd.DataFrame(adoption_rates, index=t, )

        return rates_df
    
    xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
    'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_1': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_1, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_2': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_2, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_3': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_3, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_4': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_4, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_5': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_5, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_6': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_6, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_7': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_7, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_8': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_8, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_9': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_9, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_10': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_10, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_11': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_11, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_12': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_12, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_13': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_13, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_14': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_14, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_15': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_15, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_16': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_16, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_17': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_17, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_18': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_18, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_19': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_19, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_20': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_20, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_21': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_21, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_22': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_22, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_23': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_23, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_24': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_24, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_25': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_25, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_26': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_26, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_27': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_27, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_28': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_28, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_29': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_29, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_30': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_30, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_31': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_31, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_32': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_32, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_33': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_33, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_34': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_34, 
        'xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_35': xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_35
    }
    xǁMultiProductDiffusionModelǁpredict_adoption_rate__mutmut_orig.__name__ = 'xǁMultiProductDiffusionModelǁpredict_adoption_rate'

    @property
    def param_names(self) -> Sequence[str]:
        return ["p", "Q", "m"]

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        return {}

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        return {}
