from innovate.base.base import DiffusionModel
from typing import Sequence, Dict, List
from innovate.backend import current_backend as B

class HierarchicalModel(DiffusionModel):
    def __init__(self, model, groups):
        self.model = model
        self.groups = groups
        self._params = {}

    def fit(self, t, y):
        pass

    def predict(self, t, covariates=None):
        pass

    def score(self, t: Sequence[float], y: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0