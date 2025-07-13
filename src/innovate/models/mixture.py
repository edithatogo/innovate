from ..base import DiffusionModel
from typing import Sequence, Dict, List

class MixtureModel(DiffusionModel):
    """
    Base class for mixture models.
    """
    def __init__(self, models: List[DiffusionModel], weights: List[float]):
        if len(models) != len(weights):
            raise ValueError("Number of models and weights must be the same.")

        self.models = models
        self.weights = weights
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for i, model in enumerate(self.models):
            for param_name in model.param_names:
                names.append(f"model_{i}_{param_name}")
        names.extend([f"weight_{i}" for i in range(len(self.weights))])
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        for i, model in enumerate(self.models):
            model_guesses = model.initial_guesses(t, y)
            for param_name, value in model_guesses.items():
                guesses[f"model_{i}_{param_name}"] = value
        for i, weight in enumerate(self.weights):
            guesses[f"weight_{i}"] = weight
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        for i, model in enumerate(self.models):
            model_bounds = model.bounds(t, y)
            for param_name, value in model_bounds.items():
                bounds[f"model_{i}_{param_name}"] = value
        for i in range(len(self.weights)):
            bounds[f"weight_{i}"] = (0, 1)
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = 0
        for i, model in enumerate(self.models):
            model_params = {}
            for param_name in model.param_names:
                model_params[param_name] = self._params[f"model_{i}_{param_name}"]
            model.params_ = model_params
            y_pred += self.weights[i] * model.predict(t, covariates)
        return y_pred

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    @property
    def params_(self) -> Dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: Dict[str, float]):
        self._params = value

    def predict_adoption_rate(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        rate_pred = 0
        for i, model in enumerate(self.models):
            model_params = {}
            for param_name in model.param_names:
                model_params[param_name] = self._params[f"model_{i}_{param_name}"]
            model.params_ = model_params
            rate_pred += self.weights[i] * model.predict_adoption_rate(t, covariates)
        return rate_pred
