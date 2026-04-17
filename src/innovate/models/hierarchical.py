from collections.abc import Sequence

import numpy as np

from innovate.backend import current_backend as B
from innovate.base.base import DiffusionModel
from innovate.fitters.diagnostics_contract import UncertaintySummary

from .advanced import AdvancedDiffusionModel, AdvancedModelSummary


class HierarchicalModel(AdvancedDiffusionModel):
    """Simple hierarchical wrapper to combine group-level models."""

    def __init__(self, model: DiffusionModel, groups: Sequence[str]):
        self.template = model
        self.groups = list(groups)
        self._params: dict[str, float] = {}

    # ------------------------------------------------------------------
    # DiffusionModel API helpers
    # ------------------------------------------------------------------
    @property
    def param_names(self) -> Sequence[str]:
        names: list[str] = [f"global_{p}" for p in self.template.param_names]
        for g in self.groups:
            for p in self.template.param_names:
                names.append(f"{g}_{p}")
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        """Return starting values for global and group-level parameters."""
        guesses: dict[str, float] = {}
        base = self.template.initial_guesses(t, y)
        for p, v in base.items():
            guesses[f"global_{p}"] = v
            for g in self.groups:
                guesses[f"{g}_{p}"] = 0.0
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds: dict[str, tuple] = {}
        base = self.template.bounds(t, y)
        for p, bnd in base.items():
            bounds[f"global_{p}"] = bnd
            for g in self.groups:
                bounds[f"{g}_{p}"] = bnd
        return bounds

    def fit(self, t: Sequence[float], y):
        """Fit group-level models using ScipyFitter.

        Parameters
        ----------
        t : sequence of float
            Time points.
        y : sequence or mapping
            If a dictionary is provided, it should map each group name to its
            observed series. Otherwise the same observations are used for all
            groups.
        """
        from innovate.fitters.scipy_fitter import ScipyFitter

        fitter = ScipyFitter()
        params: dict[str, float] = {}

        if isinstance(y, dict):
            for g in self.groups:
                series = y[g]
                m = self.template.__class__()
                fitter.fit(m, t, series)
                for p, val in m.params_.items():
                    params[f"{g}_{p}"] = val
            for p in self.template.param_names:
                vals = [params[f"{g}_{p}"] for g in self.groups]
                params[f"global_{p}"] = float(B.mean(B.array(vals)))
        else:
            m = self.template.__class__()
            fitter.fit(m, t, y)
            for p, val in m.params_.items():
                params[f"global_{p}"] = val
                for g in self.groups:
                    params[f"{g}_{p}"] = 0.0

        self._params = params
        return self

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        total = B.zeros(len(t))
        for g in self.groups:
            m = self.template.__class__()
            group_params = {}
            for p in self.template.param_names:
                base = self._params.get(f"global_{p}", 0.0)
                adj = self._params.get(f"{g}_{p}", 0.0)
                group_params[p] = base + adj
            m.params_ = group_params
            total += B.array(m.predict(t, covariates))
        return total

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = value

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ):
        import numpy as np

        cumulative = self.predict(t, covariates)
        rates = np.diff(B.array(cumulative), n=1)
        return np.concatenate([[rates[0]], rates])

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        y_pred = self.predict(t, covariates)
        ss_res = B.sum((B.array(y) - y_pred) ** 2)
        ss_tot = B.sum((B.array(y) - B.mean(B.array(y))) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def simulate(
        self,
        t: Sequence[float],
        n_draws: int = 1,
        random_state: int | None = None,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        """Simulate cumulative paths around the hierarchical forecast."""
        prediction = self.predict(t)
        return self._simulate_from_prediction(
            t,
            prediction,
            n_draws=n_draws,
            random_state=random_state,
            noise_scale=noise_scale,
        )

    def summarize(self, t: Sequence[float] | None = None) -> AdvancedModelSummary:
        """Return a structured summary of the hierarchical fit."""
        group_params = {
            group: {name: self._params.get(f"{group}_{name}") for name in self.template.param_names}
            for group in self.groups
        }
        return self._summary(
            family="hierarchical",
            model_name=self.__class__.__name__,
            t=t,
            provenance="deterministic",
            uncertainty=UncertaintySummary.point_estimate(
                provenance="deterministic",
                note="Hierarchical wrapper uses pooled point estimates.",
            ),
            notes=("grouped diffusion workflow",),
            details={
                "groups": list(self.groups),
                "template_model": self.template.__class__.__name__,
                "group_parameters": group_params,
            },
        )

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        """HierarchicalModel has no direct differential equation."""
        raise NotImplementedError
