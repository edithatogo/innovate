"""Tests for hype models - delayed_hype_bass and hype_modified_bass."""

import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.bass import BassModel
from innovate.hype.delayed_hype_bass import DelayedHypeBassModel
from innovate.hype.hype_cycle import HypeCycleModel
from innovate.hype.hype_modified_bass import HypeModifiedBassModel


class TestDelayedHypeBassModel:
    """Test DelayedHypeBassModel."""

    def test_init_requires_bass_and_hype(self):
        """Test initialization requires bass_model, hype_model, and delay."""
        bass = BassModel()
        hype = HypeCycleModel()
        model = DelayedHypeBassModel(bass, hype, delay=1.0)
        assert model.bass_model is bass
        assert model.hype_model is hype
        assert model.delay == 1.0

    def test_predict_unfitted_raises(self):
        """Test predict raises if inner models not fitted."""
        bass = BassModel()
        hype = HypeCycleModel()
        model = DelayedHypeBassModel(bass, hype, delay=1.0)
        with pytest.raises(RuntimeError, match="parameters set"):
            model.predict([1.0, 2.0], y0=0.0)


class TestHypeModifiedBassModel:
    """Test HypeModifiedBassModel."""

    def test_init_requires_bass_and_hype(self):
        """Test initialization requires bass_model and hype_model."""
        bass = BassModel()
        hype = HypeCycleModel()
        model = HypeModifiedBassModel(bass, hype)
        assert model.bass_model is bass
        assert model.hype_model is hype

    def test_predict_unfitted_raises(self):
        """Test predict raises if inner models not fitted."""
        bass = BassModel()
        hype = HypeCycleModel()
        model = HypeModifiedBassModel(bass, hype)
        with pytest.raises((RuntimeError, TypeError)):
            model.predict([1.0, 2.0], y0=0.0)
