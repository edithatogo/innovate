from innovate.base.base import DiffusionModel
from innovate.backend import current_backend as B
from typing import Sequence, Dict, List

class MixtureModel(DiffusionModel):
    """Combine multiple diffusion models using weighted averaging."""
    def __init__(self, models: Sequence[DiffusionModel], weights: Sequence[float] = None):
        self.models = list(models)
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        if len(weights) != len(self.models):
            raise ValueError("Length of weights must match number of models")
        self.weights = B.array(weights)
        self._params: Dict[str, float] = {}

    def fit(self, t: Sequence[float], y: Sequence[float]):
        raise NotImplementedError

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        preds: List[B.array] = []
        for idx, model in enumerate(self.models):
            param_prefix = f"model_{idx}_"
            params = {key[len(param_prefix):]: val for key, val in self._params.items() if key.startswith(param_prefix)}
            m = type(model)()  # assume default constructor works
            m.params_ = params
            preds.append(B.array(m.predict(t)))
        stacked = B.stack(preds)
        weighted = B.matmul(self.weights, stacked)
        return weighted

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
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

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        raise NotImplementedError

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for idx, model in enumerate(self.models):
            for p in model.param_names:
                names.append(f"model_{idx}_{p}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        for idx, model in enumerate(self.models):
            sub_guesses = model.initial_guesses(t, y)
            for k, v in sub_guesses.items():
                guesses[f"model_{idx}_{k}"] = v
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bnds = {}
        for idx, model in enumerate(self.models):
            sub_bounds = model.bounds(t, y)
            for k, v in sub_bounds.items():
                bnds[f"model_{idx}_{k}"] = v
        return bnds
