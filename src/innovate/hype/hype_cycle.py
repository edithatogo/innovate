# src/innovate/hype/hype_cycle.py

from typing import Dict, Sequence

from numpy import ndarray
from numpy import array, clip, exp, inf


class HypeCycleModel:
    """A model for generating a Hype Cycle curve.

    This model combines a logistic growth curve for the underlying technology
    maturity with a hype function to model the visibility of the technology.
    """

    def __init__(self):
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        return [
            "k",  # Growth rate of the logistic curve
            "t0",  # Midpoint of the logistic curve
            "a_hype",  # Amplitude of the hype
            "t_hype",  # Time of the peak of the hype
            "w_hype",  # Width of the hype
            "a_d",  # Amplitude of the disillusionment
            "t_d",  # Time of the trough of disillusionment
            "w_d",  # Width of the disillusionment
        ]

    def predict(self, t: ndarray) -> ndarray:
        """Generates the Hype Cycle curve.

        Args:
        ----
            t: A sequence of time points.

        Returns:
        -------
            The visibility of the technology at each time point.
        """
        if not self._params:
            raise RuntimeError("Model parameters have not been set.")

        k: float = self._params["k"]
        t0: float = self._params["t0"]
        a_hype: float = self._params["a_hype"]
        t_hype: float = self._params["t_hype"]
        w_hype: float = self._params["w_hype"]
        a_d: float = self._params["a_d"]
        t_d: float = self._params["t_d"]
        w_d: float = self._params["w_d"]

        t_arr: ndarray = array(t)

        # Logistic curve for technology maturity, scaled to have less impact
        maturity: ndarray = 0.5 / (1 + exp(-k * (t_arr - t0)))

        # Hype function (a combination of two Gaussians)
        hype: ndarray = a_hype * exp(-((t_arr - t_hype) ** 2) / (2 * w_hype**2))
        disillusionment: ndarray = a_d * exp(-((t_arr - t_d) ** 2) / (2 * w_d**2))

        visibility: ndarray = maturity + hype - disillusionment
        return clip(visibility, 0, inf)

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value
