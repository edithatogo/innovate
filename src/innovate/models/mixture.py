from typing import Sequence, List, Dict
import numpy as np
from innovate.base.base import DiffusionModel
from innovate.utils.metrics import calculate_r_squared

class MixtureModel(DiffusionModel):
    def __init__(self, models: List[DiffusionModel], weights: Sequence[float]):
        self.models = models
        self.weights = weights
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for idx, model in enumerate(self.models):
            for pname in model.param_names:
                names.append(f"model_{idx}_{pname}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        for idx, model in enumerate(self.models):
            for name, val in model.initial_guesses(t, y).items():
                guesses[f"model_{idx}_{name}"] = val
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        for idx, model in enumerate(self.models):
            for name, bnd in model.bounds(t, y).items():
                bounds[f"model_{idx}_{name}"] = bnd
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        preds = np.zeros_like(t, dtype=float)
        for w, m in zip(self.weights, self.models):
            preds += w * m.predict(t)
        return preds

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        y_pred = self.predict(t, covariates)
        return calculate_r_squared(y, y_pred)

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        raise NotImplementedError

    @staticmethod
    def differential_equation(y, t, p):
        raise NotImplementedError
