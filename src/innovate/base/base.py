from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeVar

# Define a type variable for the class itself, for type hinting Self
Self = TypeVar("Self", bound="DiffusionModel")


class DiffusionModel(ABC):
    """Abstract base class for all diffusion models."""

    @abstractmethod
    def predict(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts adoption levels for given time points.
        
        Parameters
        ----------
        t : Sequence[float]
            Sequence of time points for prediction
        
        Returns
        -------
        Sequence[float]
            Predicted adoption levels at each time point
            
        Raises
        ------
        RuntimeError
            If model is not fitted (params_ is empty)
        ValueError
            If time points are invalid (e.g., negative, non-numeric)
        """

    @abstractmethod
    def score(self, t: Sequence[float], y: Sequence[float]) -> float:
        """Returns the R^2 score of the model fit."""

    @property
    @abstractmethod
    def params_(self) -> dict[str, float]:
        """Returns a dictionary of fitted model parameters."""

    @params_.setter
    @abstractmethod
    def params_(self, value: dict[str, float]):
        """Sets the model parameters."""

    @abstractmethod
    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        """Predicts the rate of adoption (new adoptions per unit of time)."""

    @property
    @abstractmethod
    def param_names(self) -> Sequence[str]:
        """Returns the names of the model parameters."""

    @abstractmethod
    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Returns initial guesses for the model parameters."""

    @abstractmethod
    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        """Returns bounds for the model parameters."""

    @staticmethod
    @abstractmethod
    def differential_equation(y: float, t: float, p: Sequence[float]) -> float:
        """Returns the differential equation for the model."""

    def fit(
        self: Self,
        fitter: Any,
        t: Sequence[float],
        y: Sequence[float],
        **kwargs,
    ) -> Self:
        """Fits the diffusion model to the given time and adoption data."""
        p0 = self.initial_guesses(t, y)
        bounds = self.bounds(t, y)
        return fitter.fit(
            self,
            t,
            y,
            p0=list(p0.values()),
            bounds=list(zip(*bounds.values())),
            **kwargs,
        )
