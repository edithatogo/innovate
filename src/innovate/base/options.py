from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class FitOptions:
    """Configuration options for model fitting."""

    p0: Sequence[float] | None = None
    bounds: tuple[Sequence[float], Sequence[float]] | None = None
    weights: Sequence[float] | None = None
