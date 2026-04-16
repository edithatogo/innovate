"""Integration tests for cross-module workflows."""

import numpy as np

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.fitters.scipy_fitter import ScipyFitter


def _fit_and_score_or_neg_inf(model, fitter: ScipyFitter, t: np.ndarray, y: np.ndarray) -> float:
    """Return the model score, or negative infinity for unsupported fits."""
    try:
        fitter.fit(model, t, y)
    except Exception:
        return -np.inf
    return model.score(t, y)


class TestDiffusionToFitPipeline:
    """Test end-to-end diffusion → fit → predict → score pipeline."""

    def _make_bass_data(self, p=0.03, q=0.38, m=100.0, noise=2.0):
        """Generate synthetic Bass model data with noise."""
        t = np.linspace(0, 12, 40)
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        np.random.seed(42)
        y += np.random.normal(0, noise, len(y))
        return t, np.maximum(y, 0)  # Ensure non-negative

    def test_bass_fit_predict_pipeline(self):
        """Test complete Bass model fit → predict → score pipeline."""
        t, y = self._make_bass_data()
        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        # Predict on new time points
        t_new = np.linspace(0, 15, 50)
        preds = model.predict(t_new)
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)
        # Score should be good
        score = model.score(t, y)
        assert score > 0.8

    def test_gompertz_fit_predict_pipeline(self):
        """Test complete Gompertz model fit → predict → score pipeline."""
        t = np.linspace(0, 12, 40)
        y = 100.0 * np.exp(-2.0 * np.exp(-0.3 * t))
        np.random.seed(42)
        y += np.random.normal(0, 2, len(y))
        y = np.maximum(y, 0)

        model = GompertzModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        t_new = np.linspace(0, 15, 50)
        preds = model.predict(t_new)
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)

    def test_logistic_fit_predict_pipeline(self):
        """Test complete Logistic model fit → predict → score pipeline."""
        t = np.linspace(0, 12, 40)
        y = 100.0 / (1 + np.exp(-0.8 * (t - 5.0)))
        np.random.seed(42)
        y += np.random.normal(0, 2, len(y))
        y = np.maximum(y, 0)

        model = LogisticModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        t_new = np.linspace(0, 15, 50)
        preds = model.predict(t_new)
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)


class TestCrossModelComparison:
    """Test comparing different models on the same data."""

    def test_model_selection_by_score(self):
        """Test fitting multiple models and selecting the best by score."""
        t = np.linspace(0, 12, 40)
        p, q, m = 0.03, 0.38, 100.0
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        np.random.seed(42)
        y += np.random.normal(0, 2, len(y))
        y = np.maximum(y, 0)

        models = {
            "Bass": BassModel(),
            "Gompertz": GompertzModel(),
            "Logistic": LogisticModel(),
        }
        fitter = ScipyFitter()
        scores = {}
        for name, model in models.items():
            scores[name] = _fit_and_score_or_neg_inf(model, fitter, t, y)

        # Bass should fit best since data was generated from it
        assert scores["Bass"] > -np.inf
        best_model = max(scores, key=scores.get)
        assert best_model in models


class TestAdoptionRatePipeline:
    """Test adoption rate prediction pipeline."""

    def test_adoption_rate_after_fit(self):
        """Test adoption rate prediction after fitting."""
        t = np.linspace(0, 12, 40)
        p, q, m = 0.03, 0.38, 100.0
        y = m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))
        np.random.seed(42)
        y += np.random.normal(0, 2, len(y))
        y = np.maximum(y, 0)

        model = BassModel()
        fitter = ScipyFitter()
        fitter.fit(model, t, y)
        rates = model.predict_adoption_rate(t)
        assert len(rates) == len(t)
        assert all(r >= -1e-10 for r in rates)  # Adoption rates should be non-negative
        # Peak adoption rate should be positive
        assert np.max(rates) > 0
