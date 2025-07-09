from abc import ABC, abstractmethod
from typing import Dict, Sequence, TypeVar, Any

# Define a type variable for the class itself, for type hinting Self
Self = TypeVar('Self')

class DiffusionModel(ABC):
    """Abstract base class for all diffusion models."""

    def fit(self, fitter: Any, t: Sequence[float], y: Sequence[float], **kwargs) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        fitter.fit(self, t, y, p0=list(p0.values()), bounds=list(zip(*bounds.values())), **kwargs)
        return self

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

    @params_.setter
    @abstractmethod
    def params_(self, value: Dict[str, float]):
        """Sets the model parameters."""
        pass

    @abstractmethod
    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of adoption (new adoptions per unit of time)."""
        pass

    @property
    @abstractmethod
    def param_names(self) -> Sequence[str]:
        """Returns the names of the model parameters."""
        pass

    @abstractmethod
    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        """Returns initial guesses for the model parameters."""
        pass

    @abstractmethod
    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        """Returns bounds for the model parameters."""
        pass
