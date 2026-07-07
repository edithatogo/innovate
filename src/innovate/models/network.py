"""Network-aware diffusion model wrappers."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from innovate.base.base import DiffusionModel
from innovate.fitters.diagnostics_contract import UncertaintySummary
from innovate.fitters.scipy_fitter import ScipyFitter

from .advanced import AdvancedDiffusionModel, AdvancedModelSummary
from .contracts import NetworkDiffusionInputs


class NetworkDiffusionModel(AdvancedDiffusionModel):
    """Diffusion model that adds network spillover to node-level forecasts."""

    def __init__(
        self,
        network_inputs: NetworkDiffusionInputs,
        base_model: DiffusionModel | None = None,
        spillover_strength: float = 0.15,
    ) -> None:
        if base_model is None:
            from innovate.diffuse.bass import BassModel

            base_model = BassModel()

        self.network_inputs = network_inputs
        self.base_model = base_model
        self.spillover_strength = float(spillover_strength)
        self._params: dict[str, float] = {}
        self._node_models: list[DiffusionModel] = []
        self._fit_time = np.array([])
        self._y_orientation = "node_time"

    @property
    def param_names(self) -> Sequence[str]:
        base_names = self.base_model.param_names
        names = ["spillover_strength"]
        names.extend(
            f"node_{node_index}_{param_name}"
            for node_index, _ in enumerate(self.network_inputs.node_labels)
            for param_name in base_names
        )
        return names

    def initial_guesses(self, t: Sequence[float], y: Sequence[float]) -> dict[str, float]:
        guesses = {"spillover_strength": self.spillover_strength}
        base_guesses = self.base_model.initial_guesses(t, y)
        guesses.update(
            (f"node_{node_index}_{name}", value)
            for node_index, _ in enumerate(self.network_inputs.node_labels)
            for name, value in base_guesses.items()
        )
        return guesses

    def bounds(self, t: Sequence[float], y: Sequence[float]) -> dict[str, tuple]:
        bounds = {"spillover_strength": (0.0, np.inf)}
        base_bounds = self.base_model.bounds(t, y)
        bounds.update(
            (f"node_{node_index}_{name}", value)
            for node_index, _ in enumerate(self.network_inputs.node_labels)
            for name, value in base_bounds.items()
        )
        return bounds

    def _normalize_observations(self, t: Sequence[float], y: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        t_len = len(np.asarray(t, dtype=float))
        node_count = len(self.network_inputs.node_labels)

        if y_arr.ndim != 2:
            raise ValueError("Network observations must be a 2D array")
        if y_arr.shape == (node_count, t_len):
            self._y_orientation = "node_time"
            return y_arr
        if y_arr.shape == (t_len, node_count):
            self._y_orientation = "time_node"
            return y_arr.T
        raise ValueError("Network observations must align with the node and time dimensions")

    def fit(self, t: Sequence[float], y: Sequence[Sequence[float]] | np.ndarray):
        t_arr = np.asarray(t, dtype=float)
        y_arr = self._normalize_observations(t_arr, y)
        if y_arr.shape[0] != len(self.network_inputs.node_labels):
            raise ValueError("Observation rows must match the number of nodes")

        fitter = ScipyFitter()
        self._node_models = []
        for node_series in y_arr:
            model = copy.deepcopy(self.base_model)
            fitter.fit(model, t_arr, node_series)
            self._node_models.append(model)

        self._fit_time = t_arr
        self.spillover_strength = float(self.spillover_strength)
        self._params = {"spillover_strength": self.spillover_strength}
        for node_index, model in enumerate(self._node_models):
            for name, value in model.params_.items():
                self._params[f"node_{node_index}_{name}"] = float(value)
        return self

    def _base_predictions(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        if not self._node_models:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        t_arr = np.asarray(t, dtype=float)
        return np.vstack([np.asarray(model.predict(t_arr, covariates), dtype=float) for model in self._node_models])

    def predict(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        base_predictions = self._base_predictions(t, covariates)
        spillover = self.network_inputs.row_normalized_adjacency() @ base_predictions
        adjusted = base_predictions + self.spillover_strength * spillover
        return np.maximum.accumulate(np.maximum(adjusted, 0.0), axis=1)

    @property
    def params_(self) -> dict[str, float]:
        return self._params

    @params_.setter
    def params_(self, value: dict[str, float]):
        self._params = dict(value)
        self.spillover_strength = float(value.get("spillover_strength", self.spillover_strength))

        grouped_params: list[dict[str, float]] = [{} for _ in self.network_inputs.node_labels]
        for key, param_value in value.items():
            if key.startswith("node_"):
                parts = key.split("_", 2)
                if len(parts) >= 3 and parts[1].isdigit():
                    idx = int(parts[1])
                    if idx < len(grouped_params):
                        grouped_params[idx][parts[2]] = float(param_value)

        node_models: list[DiffusionModel] = []
        for node_params in grouped_params:
            model = copy.deepcopy(self.base_model)
            if node_params:
                model.params_ = node_params
            node_models.append(model)
        self._node_models = node_models

    def score(
        self,
        t: Sequence[float],
        y: Sequence[Sequence[float]] | np.ndarray,
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> float:
        y_arr = self._normalize_observations(t, y)
        y_pred = self.predict(t, covariates)
        ss_res = float(np.sum((y_arr - y_pred) ** 2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr, axis=1, keepdims=True)) ** 2))
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict_adoption_rate(
        self,
        t: Sequence[float],
        covariates: dict[str, Sequence[float]] | None = None,
    ) -> np.ndarray:
        prediction = self.predict(t, covariates)
        return np.diff(prediction, axis=1, prepend=prediction[:, :1])

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
        adjacency = np.asarray(self.network_inputs.adjacency, dtype=float)
        possible_edges = max(1, adjacency.shape[0] * (adjacency.shape[0] - 1))
        edge_count = int(np.count_nonzero(adjacency))
        degree = adjacency.sum(axis=1)

        node_count = len(self.network_inputs.node_labels)
        return self._summary(
            family="network_diffusion",
            model_name=self.__class__.__name__,
            t=t,
            provenance="deterministic",
            uncertainty=UncertaintySummary.point_estimate(
                provenance="deterministic",
                note="Network spillover is modeled with row-normalized peer influence.",
            ),
            notes=("network contagion and spillover workflow",),
            details={
                "node_count": node_count,
                "edge_count": edge_count,
                "edge_density": edge_count / possible_edges,
                "average_degree": float(np.mean(degree)) if degree.size else 0.0,
                "spillover_strength": self.spillover_strength,
                "node_labels": list(self.network_inputs.node_labels),
            },
        )

    def set_intervention_nodes(self, node_indices: list[int]) -> None:
        """Mark specific nodes as intervention targets.

        Intervention nodes receive an additional adoption boost during
        prediction, simulating targeted policy or marketing interventions.

        Parameters
        ----------
        node_indices : list of int
            Indices of nodes to mark as intervention targets.
        """
        self._intervention_nodes = list(node_indices)

    def _has_intervention_nodes(self) -> bool:
        return hasattr(self, "_intervention_nodes") and bool(self._intervention_nodes)

    @staticmethod
    def differential_equation(t, y, params, covariates, t_eval):
        raise NotImplementedError
