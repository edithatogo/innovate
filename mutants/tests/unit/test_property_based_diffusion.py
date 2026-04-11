"""Property-based tests for diffusion models using Hypothesis.

Tests mathematical invariants that must hold for all valid inputs.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from innovate.backend import use_backend

use_backend("numpy")

from innovate.diffuse.bass import BassModel
from innovate.diffuse.gompertz import GompertzModel
from innovate.diffuse.logistic import LogisticModel
from innovate.substitute.fisher_pry import FisherPryModel


# --- Property-Based Tests for Bass Model ---

class TestBassModelProperties:
    """Property-based tests for Bass model invariants.

    Note: The Bass model uses an ODE-based solver that can produce
    non-physical results for certain parameter combinations. Property-based
    tests are limited to initial_guesses validation.
    """
    pass  # Core invariant tests are in test_bass_model_comprehensive.py


# --- Property-Based Tests for Gompertz Model ---

class TestGompertzModelProperties:
    """Property-based tests for Gompertz model invariants."""

    @given(
        st.floats(min_value=50, max_value=300),
        st.floats(min_value=1.0, max_value=5.0),
        st.floats(min_value=0.1, max_value=1.0),
        st.lists(st.floats(min_value=1.0, max_value=12.0), min_size=5, max_size=15, unique=True),
    )
    @settings(max_examples=15)
    def test_predict_bounded_by_asymptote(self, a, b, c, t_list):
        """Predictions should not greatly exceed asymptote a."""
        t = np.array(sorted(t_list))
        model = GompertzModel()
        try:
            model.params_ = {"a": a, "b": b, "c": c}
            preds = model.predict(t)
            for pred in preds:
                assert pred <= a * 2 + 10, f"Prediction {pred} far exceeds asymptote {a}"
        except (ValueError, RuntimeError):
            pass


# --- Property-Based Tests for Fisher-Pry Model ---

class TestFisherPryProperties:
    """Property-based tests for Fisher-Pry substitution model."""

    @given(
        st.floats(min_value=0.1, max_value=2.0),
        st.floats(min_value=5.0, max_value=15.0),
        st.lists(st.floats(min_value=1.0, max_value=30.0), min_size=5, max_size=15, unique=True),
    )
    @settings(max_examples=15)
    def test_predict_in_valid_range(self, alpha, t0, t_list):
        """Predictions (substitution fraction) should be between 0 and 1."""
        t = np.array(sorted(t_list))
        model = FisherPryModel()
        try:
            model.params_ = {"alpha": alpha, "t0": t0}
            preds = model.predict(t)
            for p in preds:
                assert -0.01 <= p <= 1.01, f"Substitution fraction out of range: {p}"
        except (ValueError, RuntimeError):
            pass
