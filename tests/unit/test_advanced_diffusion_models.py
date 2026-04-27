"""Tests for canonical advanced diffusion inference workflows."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

import innovate
from innovate.capabilities import ModelCapability
from innovate.diffuse.bass import BassModel
from innovate.diffuse.logistic import LogisticModel
from innovate.models.advanced import AdvancedDiffusionModel, AdvancedModelSummary
from innovate.models.hierarchical import HierarchicalModel
from innovate.models.mixture import MixtureModel


def _bass_series(t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
    exp_term = np.exp(-(p + q) * t)
    return m * (1 - exp_term) / (1 + (q / p) * exp_term)


def test_canonical_advanced_exports_and_registry_metadata():
    """Advanced workflows should be importable from canonical package locations."""
    assert innovate.models.AdvancedDiffusionModel is AdvancedDiffusionModel
    assert innovate.models.AdvancedModelSummary is AdvancedModelSummary
    assert innovate.models.HierarchicalModel is HierarchicalModel
    assert innovate.models.MixtureModel is MixtureModel
    assert innovate.HierarchicalModel is HierarchicalModel
    assert innovate.MixtureModel is MixtureModel

    registry = innovate.get_model_registry()
    assert isinstance(registry["hierarchical"], ModelCapability)
    assert registry["hierarchical"].stability == "experimental"
    assert registry["hierarchical"].supports_simulation is True
    assert registry["hierarchical"].supports_summarize is True
    assert registry["regime_switching"].optional_dependencies == ("ruptures",)
    assert innovate.get_model_capability("latent_process").import_path.endswith("LatentProcessDiffusionModel")


def test_hierarchical_model_exposes_simulate_and_summarize():
    """The hierarchical workflow should expose a consistent advanced-model surface."""
    model = HierarchicalModel(BassModel(), ["alpha", "beta"])
    model.params_ = {
        "global_p": 0.02,
        "global_q": 0.2,
        "global_m": 900.0,
        "alpha_p": 0.01,
        "alpha_q": 0.03,
        "alpha_m": 120.0,
        "beta_p": -0.002,
        "beta_q": 0.01,
        "beta_m": -40.0,
    }

    t = np.linspace(1, 20, 20)
    prediction = model.predict(t)
    simulated = model.simulate(t, n_draws=3, random_state=7)
    summary = model.summarize(t)

    assert prediction.shape == (20,)
    assert simulated.shape == (3, 20)
    assert isinstance(summary, AdvancedModelSummary)
    assert summary.family == "hierarchical"
    assert summary.uncertainty.report_type == "point_estimate"
    assert summary.details["groups"] == ["alpha", "beta"]
    assert summary.forecast.shape == (20,)


def test_mixture_model_exposes_simulate_and_summarize():
    """The mixture workflow should expose a consistent advanced-model surface."""
    models = [LogisticModel(), LogisticModel()]
    model = MixtureModel(models, [0.55, 0.45])
    model.params_ = {
        "model_0_L": 80.0,
        "model_0_k": 0.1,
        "model_0_x0": 4.0,
        "model_1_L": 120.0,
        "model_1_k": 0.15,
        "model_1_x0": 7.5,
        "weight_0": 0.55,
        "weight_1": 0.45,
    }

    t = np.linspace(1, 20, 20)
    prediction = model.predict(t)
    simulated = model.simulate(t, n_draws=2, random_state=13)
    summary = model.summarize(t)

    assert prediction.shape == (20,)
    assert simulated.shape == (2, 20)
    assert isinstance(summary, AdvancedModelSummary)
    assert summary.family == "mixture"
    assert summary.details["num_components"] == 2
    assert summary.details["component_weights"]["component_0"] == pytest.approx(0.55)
    assert summary.forecast.shape == (20,)


@pytest.mark.skipif(find_spec("ruptures") is None, reason="ruptures is required for the regime-switching workflow")
def test_latent_process_and_regime_switching_workflows_fit_predict_simulate():
    """The new advanced workflows should fit, predict, simulate, and summarize."""
    from innovate.models.advanced import LatentProcessDiffusionModel, RegimeSwitchingDiffusionModel

    t = np.arange(1, 41, dtype=float)
    base = _bass_series(t, 0.02, 0.22, 1000.0)
    latent_trend = np.where(t < 20, -15.0 + 0.25 * t, 20.0 + 0.45 * (t - 20))
    y = np.maximum.accumulate(np.maximum(base + latent_trend, 0.0))

    latent = LatentProcessDiffusionModel(BassModel(), smoothing=0.35)
    latent.fit(t, y)
    latent_prediction = latent.predict(t)
    latent_draws = latent.simulate(t, n_draws=4, random_state=11)
    latent_summary = latent.summarize(t)

    assert latent_prediction.shape == (40,)
    assert latent_draws.shape == (4, 40)
    assert latent_summary.family == "latent_process"
    assert latent_summary.details["latent_state_length"] == 40
    assert latent_summary.uncertainty.report_type == "point_estimate"

    regime = RegimeSwitchingDiffusionModel(BassModel())
    regime.fit(t, y)
    regime_prediction = regime.predict(t)
    regime_draws = regime.simulate(t, n_draws=2, random_state=19)
    regime_summary = regime.summarize(t)

    assert regime_prediction.shape == (40,)
    assert regime_draws.shape == (2, 40)
    assert regime_summary.family == "regime_switching"
    assert regime_summary.details["regime_count"] >= 1
    assert regime_summary.details["changepoint_index"] >= 0
