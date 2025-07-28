from innovate.base.base import DiffusionModel
from typing import Sequence, Dict, List
from innovate.backend import current_backend as B

class HierarchicalModel(DiffusionModel):
    """Simple hierarchical model with global and group-level parameters."""
    def __init__(self, model: DiffusionModel, groups: Sequence[str]):
        self.model = model
        self.groups = list(groups)
        self._params: Dict[str, float] = {}

    def fit(self, t: Sequence[float], y: Sequence[float]):
        raise NotImplementedError

    def _group_params(self, group: str) -> Dict[str, float]:
        params = {}
        for name in self.model.param_names:
            global_key = f"global_{name}"
            group_key = f"{group}_{name}"
            params[name] = self._params.get(global_key, 0.0) + self._params.get(group_key, 0.0)
        return params

    def predict(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        preds: List[B.array] = []
        for group in self.groups:
            params = self._group_params(group)
            m = type(self.model)()
            m.params_ = params
            preds.append(B.array(m.predict(t)))
        stacked = B.stack(preds)
        mean_pred = B.sum(stacked, axis=0) / len(preds)
        return mean_pred

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

    def predict_adoption_rate(self, t: Sequence[float]) -> Sequence[float]:
        raise NotImplementedError

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        raise NotImplementedError

    @property
    def param_names(self) -> Sequence[str]:
        names = []
        for base in self.model.param_names:
            names.append(f"global_{base}")
            for g in self.groups:
                names.append(f"{g}_{base}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses = {}
        base_guesses = self.model.initial_guesses(t, y)
        for name, val in base_guesses.items():
            guesses[f"global_{name}"] = val
            for g in self.groups:
                guesses[f"{g}_{name}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bnds = {}
        base_bounds = self.model.bounds(t, y)
        for name, val in base_bounds.items():
            bnds[f"global_{name}"] = val
            for g in self.groups:
                bnds[f"{g}_{name}"] = val
        return bnds
