"""Policy-timing and hazard-style diffusion model wrappers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.fitters.diagnostics_contract import UncertaintySummary
from innovate.fitters.scipy_fitter import ScipyFitter

from .advanced import AdvancedDiffusionModel, AdvancedModelSummary
from .contracts import PolicyTimingInputs


class PolicyHazardDiffusionModel(AdvancedDiffusionModel):
    """Diffusion model that applies event-history style policy timing effects."""

    def __init__(
        self,
        policy_inputs: PolicyTimingInputs | None = None,
        base_model: DiffusionModel | None = None,
        decay: float = 0.5,
    ) -> None:
        if base_model is None:
            from innovate.diffuse.bass import BassModel

            base_model = BassModel()

        self.policy_inputs = policy_inputs or PolicyTimingInputs(event_times=(), event_effects=(), event_labels=())
        self.base_model = base_model
        self.decay = float(decay)
        self._params: dict[str, float] = {}

    @property
    def param_names(self) -> Sequence[str]:
        names = ["decay"]
        for name in self.base_model.param_names:
            names.append(f"base_{name}")
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> dict[str, float]:
        guesses = {"decay": self.decay}
        for name, value in self.base_model.initial_guesses(t, y).items():
            guesses[f"base_{name}"] = value
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {"decay": (0.0, np.inf)}
        for name, value in self.base_model.bounds(t, y).items():
            bounds[f"base_{name}"] = value
        return bounds

    def fit(self, t: Sequence[float], y: Sequence[float]):
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 1:
            raise ValueError("Policy observations must be a 1D cumulative series")

        fitter = ScipyFitter()
        fitter.fit(self.base_model, t_arr, y_arr)

        self._params = {"decay": self.decay}
        self._params.update({f"base_{name}": float(value) for name, value in self.base_model.params_.items()})
        return self

    def _policy_multiplier(self, t: Sequence[float]) -> np.ndarray:
        return 1.0 + self.policy_inputs.effect_profile(t, self.decay)

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = np.asarray(t, dtype=float)
        baseline = np.asarray(self.base_model.predict(t_arr, covariates), dtype=float)
        adjusted = baseline * self._policy_multiplier(t_arr)
        return np.maximum.accumulate(np.maximum(adjusted, 0.0))

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = dict(value)
        self.decay = float(value.get("decay", self.decay))
        base_params = {
            key[len("base_") :]: float(param_value) for key, param_value in value.items() if key.startswith("base_")
        }
        if base_params:
            self.base_model.params_ = base_params

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        y_arr = np.asarray(y, dtype=float)
        y_pred = self.predict(t, covariates)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        prediction = self.predict(t, covariates)
        return np.diff(prediction, prepend=prediction[:1])

    def simulate(
        self,
        t: Sequence[float],
        n_draws: int = 1,
        random_state: int | None = None,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        prediction = self.predict(t)
        return self._simulate_from_prediction(
            t,
            prediction,
            n_draws=n_draws,
            random_state=random_state,
            noise_scale=noise_scale,
        )

    def summarize(self, t: Sequence[float] | None = None) -> AdvancedModelSummary:
        event_effects = np.asarray(self.policy_inputs.event_effects, dtype=float)
        return self._summary(
            family="policy_hazard",
            model_name=self.__class__.__name__,
            t=t,
            provenance="deterministic",
            uncertainty=UncertaintySummary.point_estimate(
                provenance="deterministic",
                note="Policy timing effects are applied as a decaying multiplier on the baseline forecast.",
            ),
            notes=("event-history and policy timing workflow",),
            details={
                "event_count": len(self.policy_inputs.event_times),
                "event_labels": list(self.policy_inputs.event_labels),
                "event_times": list(self.policy_inputs.event_times),
                "average_effect": float(np.mean(event_effects)) if event_effects.size else 0.0,
                "decay": self.decay,
            },
        )

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        raise NotImplementedError
