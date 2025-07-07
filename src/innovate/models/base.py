from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar

# Define a type variable for the class itself, for type hinting Self
Self = TypeVar('Self')

class DiffusionModel(ABC):
    """Abstract base class for all diffusion models."""

    @abstractmethod
    def fit(self, t: Sequence[float], y: Sequence[float]) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        pass

    @abstractmethod
    def predict(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts adoption levels for given time points."""
        pass

    @abstractmethod
    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Returns the R^2 score of the model fit."""
        pass

    @property
    @abstractmethod
    def params_(self) -> Dict[str, float]:
        """Returns a dictionary of fitted model parameters."""
        pass

    @abstractmethod
    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of adoption (new adoptions per unit of time)."""
        pass
