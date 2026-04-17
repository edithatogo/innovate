"""Canonical abstractions and helpers for advanced diffusion workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.fitters.diagnostics_contract import UncertaintySummary
from innovate.fitters.scipy_fitter import ScipyFitter
from innovate.reduce.analysis import find_changepoint


@dataclass(frozen=True, slots=True)
class AdvancedModelSummary:
    """Structured summary for an advanced diffusion workflow."""

    family: str
    model_name: str
    support_level: str = "supported"
    provenance: str = "unknown"
    parameters: dict[str, float] = field(default_factory=dict)
    uncertainty: UncertaintySummary = field(default_factory=UncertaintySummary.point_estimate)
    notes: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)
    forecast: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary to a JSON-friendly payload."""
        return {
            "family": self.family,
            "model_name": self.model_name,
            "support_level": self.support_level,
            "provenance": self.provenance,
            "parameters": self.parameters,
            "uncertainty": self.uncertainty.to_dict(),
            "notes": list(self.notes),
            "details": self.details,
            "forecast": self.forecast.tolist(),
        }


class AdvancedDiffusionModel(DiffusionModel, ABC):
    """Base class for advanced diffusion workflows with simulation helpers."""

    @abstractmethod
    def simulate(
        self,
        t: Sequence[float],
        n_draws: int = 1,
        random_state: int | None = None,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        """Draw simulated adoption trajectories from the fitted model."""

    @abstractmethod
    def summarize(self, t: Sequence[float] | None = None) -> AdvancedModelSummary:
        """Return a structured summary of the fitted workflow."""

    @staticmethod
    def _ensure_array(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def _simulate_from_prediction(
        self,
        t: Sequence[float],
        prediction: Sequence[float],
        *,
        n_draws: int = 1,
        random_state: int | None = None,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        """Create monotone cumulative draws around a forecast path."""
        pred = self._ensure_array(prediction)
        if pred.size == 0:
            return pred

        scale = float(noise_scale if noise_scale is not None else max(1.0, np.std(pred) * 0.05))
        rng = np.random.default_rng(random_state)

        draws = []
        for _ in range(max(1, int(n_draws))):
            noise = rng.normal(0.0, scale, size=pred.shape)
            draw = np.maximum.accumulate(np.maximum(pred + noise, 0.0), axis=-1)
            draws.append(draw)

        stacked = np.stack(draws)
        return stacked[0] if stacked.shape[0] == 1 else stacked

    def _summary(
        self,
        *,
        family: str,
        model_name: str,
        t: Sequence[float] | None = None,
        support_level: str = "supported",
        provenance: str = "unknown",
        uncertainty: UncertaintySummary | None = None,
        notes: Sequence[str] = (),
        details: dict[str, object] | None = None,
    ) -> AdvancedModelSummary:
        forecast = np.array([])
        if t is not None:
            forecast = self._ensure_array(self.predict(t))

        return AdvancedModelSummary(
            family=family,
            model_name=model_name,
            support_level=support_level,
            provenance=provenance,
            parameters=dict(self.params_),
            uncertainty=uncertainty or UncertaintySummary.point_estimate(provenance=provenance),
            notes=tuple(notes),
            details={} if details is None else dict(details),
            forecast=forecast,
        )


class LatentProcessDiffusionModel(AdvancedDiffusionModel):
    """State-space style diffusion wrapper with a latent residual process."""

    def __init__(self, base_model: DiffusionModel | None = None, smoothing: float = 0.3):
        if base_model is None:
            from innovate.diffuse.bass import BassModel

            base_model = BassModel()

        self.base_model = base_model
        self.smoothing = float(smoothing)
        self._params: dict[str, float] = {}
        self.latent_state_ = np.array([])
        self._fit_time = np.array([])
        self.noise_scale_ = 0.0

    @property
    def param_names(self) -> Sequence[str]:
        return [f"base_{name}" for name in self.base_model.param_names] + [
            "latent_smoothing",
            "latent_noise",
        ]

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {f"base_{name}": value for name, value in self.base_model.initial_guesses(t, y).items()}
        guesses["latent_smoothing"] = self.smoothing
        guesses["latent_noise"] = float(np.std(np.asarray(y, dtype=float)) if len(y) else 1.0)
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {f"base_{name}": value for name, value in self.base_model.bounds(t, y).items()}
        bounds["latent_smoothing"] = (0.0, 1.0)
        bounds["latent_noise"] = (0.0, np.inf)
        return bounds

    def fit(self, t: Sequence[float], y: Sequence[float]):
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        fitter = ScipyFitter()
        fitter.fit(self.base_model, t_arr, y_arr)

        baseline = np.asarray(self.base_model.predict(t_arr), dtype=float)
        residuals = y_arr - baseline

        latent = np.zeros_like(residuals)
        for index, residual in enumerate(residuals):
            if index == 0:
                latent[index] = residual
            else:
                latent[index] = self.smoothing * residual + (1.0 - self.smoothing) * latent[index - 1]

        self.latent_state_ = latent
        self._fit_time = t_arr
        self.noise_scale_ = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        self._params = {
            **{f"base_{name}": float(value) for name, value in self.base_model.params_.items()},
            "latent_smoothing": self.smoothing,
            "latent_noise": self.noise_scale_,
        }
        return self

    def _latent_offset(self, t: Sequence[float]) -> np.ndarray:
        if self.latent_state_.size == 0 or self._fit_time.size == 0:
            return np.zeros(len(t), dtype=float)
        t_arr = np.asarray(t, dtype=float)
        return np.interp(
            t_arr,
            self._fit_time,
            self.latent_state_,
            left=self.latent_state_[0],
            right=self.latent_state_[-1],
        )

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = np.asarray(t, dtype=float)
        baseline = np.asarray(self.base_model.predict(t_arr, covariates), dtype=float)
        adjusted = np.maximum.accumulate(np.maximum(baseline + self._latent_offset(t_arr), 0.0))
        return adjusted

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = dict(value)
        base_params = {key[len("base_") :]: float(val) for key, val in value.items() if key.startswith("base_")}
        if base_params:
            self.base_model.params_ = base_params
        self.smoothing = float(value.get("latent_smoothing", self.smoothing))
        self.noise_scale_ = float(value.get("latent_noise", self.noise_scale_))

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        y_pred = np.asarray(self.predict(t, covariates), dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        predictions = np.asarray(self.predict(t, covariates), dtype=float)
        if predictions.size < 2:
            return predictions
        deltas = np.diff(predictions, prepend=predictions[0])
        return np.maximum(deltas, 0.0)

    def simulate(
        self,
        t: Sequence[float],
        n_draws: int = 1,
        random_state: int | None = None,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        prediction = self.predict(t)
        scale = noise_scale if noise_scale is not None else self.noise_scale_ or None
        return self._simulate_from_prediction(
            t,
            prediction,
            n_draws=n_draws,
            random_state=random_state,
            noise_scale=scale,
        )

    def summarize(self, t: Sequence[float] | None = None) -> AdvancedModelSummary:
        return self._summary(
            family="latent_process",
            model_name=self.__class__.__name__,
            t=t,
            provenance="deterministic",
            uncertainty=UncertaintySummary.point_estimate(
                provenance="deterministic",
                note="Latent state is estimated with an exponential-smoothing proxy.",
            ),
            notes=("state-space style latent residual process",),
            details={
                "base_model": self.base_model.__class__.__name__,
                "latent_state_length": int(self.latent_state_.size),
                "latent_smoothing": self.smoothing,
                "latent_noise": self.noise_scale_,
            },
        )

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        raise NotImplementedError


class RegimeSwitchingDiffusionModel(AdvancedDiffusionModel):
    """Piecewise diffusion model that switches regime at a detected changepoint."""

    def __init__(self, base_model: DiffusionModel | None = None):
        if base_model is None:
            from innovate.diffuse.bass import BassModel

            base_model = BassModel()

        self.base_model = base_model
        self._params: dict[str, float] = {}
        self.changepoint_index_ = -1
        self._fit_time = np.array([])
        self._regime_models: list[DiffusionModel] = []
        self.noise_scale_ = 0.0

    @property
    def param_names(self) -> Sequence[str]:
        names = ["changepoint_index"]
        for regime in range(2):
            for name in self.base_model.param_names:
                names.append(f"regime_{regime}_{name}")
        return names

    def initial_guesses(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> dict[str, float]:
        guesses = {"changepoint_index": float(max(1, len(t) // 2))}
        base_guesses = self.base_model.initial_guesses(t, y)
        for regime in range(2):
            for name, value in base_guesses.items():
                guesses[f"regime_{regime}_{name}"] = value
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {"changepoint_index": (0, max(1, len(t) - 1))}
        base_bounds = self.base_model.bounds(t, y)
        for regime in range(2):
            for name, value in base_bounds.items():
                bounds[f"regime_{regime}_{name}"] = value
        return bounds

    def fit(self, t: Sequence[float], y: Sequence[float]):
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if t_arr.size != y_arr.size:
            raise ValueError("Time and observation sequences must be the same length.")

        changepoint_index = find_changepoint(y_arr)
        if changepoint_index <= 0 or changepoint_index >= len(t_arr) - 1:
            changepoint_index = max(1, len(t_arr) // 2)

        fitter = ScipyFitter()
        self._regime_models = []
        segments = [(0, changepoint_index), (changepoint_index, len(t_arr))]

        for start, end in segments:
            if end - start < 2:
                continue
            regime_model = self.base_model.__class__()
            fitter.fit(regime_model, t_arr[start:end], y_arr[start:end])
            self._regime_models.append(regime_model)

        if not self._regime_models:
            regime_model = self.base_model.__class__()
            fitter.fit(regime_model, t_arr, y_arr)
            self._regime_models = [regime_model]
            segments = [(0, len(t_arr))]
            changepoint_index = len(t_arr) // 2

        self.changepoint_index_ = int(changepoint_index)
        self._fit_time = t_arr

        self._params = {"changepoint_index": float(self.changepoint_index_)}
        for regime_index, regime_model in enumerate(self._regime_models):
            for name, value in regime_model.params_.items():
                self._params[f"regime_{regime_index}_{name}"] = float(value)

        residuals = y_arr - np.asarray(self.predict(t_arr), dtype=float)
        self.noise_scale_ = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        return self

    def _segment_predictions(self, t: Sequence[float]) -> np.ndarray:
        t_arr = np.asarray(t, dtype=float)
        if not self._regime_models:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        if len(self._regime_models) == 1 or self.changepoint_index_ >= len(t_arr) - 1:
            return np.asarray(self._regime_models[0].predict(t_arr), dtype=float)

        split = min(max(1, self.changepoint_index_), len(t_arr) - 1)
        first = np.asarray(self._regime_models[0].predict(t_arr[:split]), dtype=float)
        second = np.asarray(self._regime_models[-1].predict(t_arr[split:]), dtype=float)

        if first.size == 0:
            return np.maximum.accumulate(np.maximum(second, 0.0))
        if second.size == 0:
            return np.maximum.accumulate(np.maximum(first, 0.0))

        offset = first[-1] - second[0]
        adjusted_second = second + offset
        return np.concatenate([first, adjusted_second])

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        if not self._params:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        prediction = self._segment_predictions(t)
        return np.maximum.accumulate(np.maximum(prediction, 0.0))

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = dict(value)
        self.changepoint_index_ = int(value.get("changepoint_index", self.changepoint_index_))

    def score(
        self,
        t: Sequence[float],
        y: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        y_pred = np.asarray(self.predict(t, covariates), dtype=float)
        y_arr = np.asarray(y, dtype=float)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> Sequence[float]:
        predictions = np.asarray(self.predict(t, covariates), dtype=float)
        if predictions.size < 2:
            return predictions
        deltas = np.diff(predictions, prepend=predictions[0])
        return np.maximum(deltas, 0.0)

    def simulate(
        self,
        t: Sequence[float],
        n_draws: int = 1,
        random_state: int | None = None,
        noise_scale: float | None = None,
    ) -> np.ndarray:
        prediction = self.predict(t)
        scale = noise_scale if noise_scale is not None else self.noise_scale_ or None
        return self._simulate_from_prediction(
            t,
            prediction,
            n_draws=n_draws,
            random_state=random_state,
            noise_scale=scale,
        )

    def summarize(self, t: Sequence[float] | None = None) -> AdvancedModelSummary:
        regime_names = [model.__class__.__name__ for model in self._regime_models]
        return self._summary(
            family="regime_switching",
            model_name=self.__class__.__name__,
            t=t,
            provenance="deterministic",
            uncertainty=UncertaintySummary.point_estimate(
                provenance="deterministic",
                note="Regime split is detected with a changepoint heuristic.",
            ),
            notes=("piecewise diffusion workflow",),
            details={
                "changepoint_index": int(self.changepoint_index_),
                "regime_count": len(self._regime_models),
                "regime_models": regime_names,
            },
        )

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        raise NotImplementedError
