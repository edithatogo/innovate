from innovate.base.base import DiffusionModel
from typing import Sequence, Dict, List
from innovate.backend import current_backend as B

class MixtureModel(DiffusionModel):
    """Combine predictions from multiple submodels using fixed weights."""

    def __init__(self, models: Sequence[DiffusionModel], weights: Sequence[float]):
        if len(models) != len(weights):
            raise ValueError("models and weights must have the same length")
        self.models = list(models)
        weights_arr = B.array(weights)
        self.weights = weights_arr / B.sum(weights_arr)
        self._params: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # DiffusionModel API
    # ------------------------------------------------------------------
    @property
    def param_names(self) -> Sequence[str]:
        names: List[str] = []
        for i, model in enumerate(self.models):
            for pname in model.param_names:
                names.append(f"model_{i}_{pname}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
        guesses: Dict[str, float] = {}
        for i, model in enumerate(self.models):
            for pn, val in model.initial_guesses(t, y).items():
                guesses[f"model_{i}_{pn}"] = val
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> Dict[str, tuple]:
        bounds: Dict[str, tuple] = {}
        for i, model in enumerate(self.models):
            for pn, val in model.bounds(t, y).items():
                bounds[f"model_{i}_{pn}"] = val
        return bounds

    def fit(self, t: Sequence[float], y: Sequence[float]):
        """Fit each submodel independently to the data using ScipyFitter."""
        from innovate.fitters.scipy_fitter import ScipyFitter

        self._params = {}
        fitter = ScipyFitter()
        for i, model in enumerate(self.models):
            fitter.fit(model, t, y)
            for pn, val in model.params_.items():
                self._params[f"model_{i}_{pn}"] = val
        return self


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

    def predict_adoption_rate(self, t: Sequence[float], covariates: Dict[str, Sequence[float]] = None) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        import numpy as np

        rates = np.diff(B.array(y_pred), n=1)
        rates = np.concatenate([[rates[0]], rates])
        return rates

    @staticmethod
    def differential_equation(y, t, p):
        raise NotImplementedError("MixtureModel does not implement a differential equation")

    @staticmethod
    def differential_equation(y, t, p):
        raise NotImplementedError

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
