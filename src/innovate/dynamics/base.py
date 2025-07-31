# src/innovate/dynamics/base.py

from abc import ABC, abstractmethod
import numpy as np


class GrowthCurve(ABC):
    """
    Abstract base class for growth curves.
    """

    @abstractmethod
    def compute_growth_rate(self, current_adopters, total_potential, **params):
        """
        Calculates the instantaneous growth rate.
        """
        pass


class ContagionSpread(ABC):
    """
    Abstract base class for contagion models.
    """

    @abstractmethod
    def differential(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Defines the differential equations for the contagion model.
        """
        pass
