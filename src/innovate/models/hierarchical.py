from ..base import DiffusionModel
from typing import Sequence, Dict, List

class HierarchicalModel(DiffusionModel):
    """
    Base class for hierarchical models.
    """
    def __init__(self, model: DiffusionModel, groups: List[str]):
        self.model = model
        self.groups = groups
        self._params: Dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for param_name in self.model.param_names:
            names.append(f"global_{param_name}")
        for group in self.groups:
            for param_name in self.model.param_names:
                names.append(f"{group}_{param_name}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        model_guesses = self.model.initial_guesses(t, y)
        for param_name, value in model_guesses.items():
            guesses[f"global_{param_name}"] = value
        for group in self.groups:
            for param_name, value in model_guesses.items():
                guesses[f"{group}_{param_name}"] = value
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds = {}
        model_bounds = self.model.bounds(t, y)
        for param_name, value in model_bounds.items():
            bounds[f"global_{param_name}"] = value
        for group in self.groups:
            for param_name, value in model_bounds.items():
                bounds[f"{group}_{param_name}"] = value
        return bounds

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        y_pred = 0
        for group in self.groups:
            model_params = {}
            for param_name in self.model.param_names:
                model_params[param_name] = self._params[f"{group}_{param_name}"]
            self.model.params_ = model_params
            y_pred += self.model.predict(t, covariates)
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
        for group in self.groups:
            model_params = {}
            for param_name in self.model.param_names:
                model_params[param_name] = self._params[f"{group}_{param_name}"]
            self.model.params_ = model_params
            rate_pred += self.model.predict_adoption_rate(t, covariates)
        return rate_pred
