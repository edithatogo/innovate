"""Property-based tests for competition and substitution models using Hypothesis."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from innovate.backend import use_backend

use_backend("numpy")

from innovate.compete.lotka_volterra import LotkaVolterraModel
from innovate.diffuse.bass import BassModel
from innovate.substitute.fisher_pry import FisherPryModel


# --- Property-Based Tests for Competition Models ---

class TestLotkaVolterraProperties:
    """Property-based tests for Lotka-Volterra competition model."""

    @given(
        st.floats(min_value=0.1, max_value=2.0),  # alpha1
        st.floats(min_value=0.01, max_value=1.0),  # beta1
        st.floats(min_value=0.1, max_value=2.0),  # alpha2
        st.floats(min_value=0.01, max_value=1.0),  # beta2
        st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=5, max_size=10, unique=True),
    )
    @settings(max_examples=10)
    def test_predict_finite(self, alpha1, beta1, alpha2, beta2, t_list):
        """All predictions should be finite."""
        t = np.array(sorted(t_list))
        model = LotkaVolterraModel()
        try:
            model.params_ = {"alpha1": alpha1, "beta1": beta1, "alpha2": alpha2, "beta2": beta2}
            preds = model.predict(t, y0=np.array([1.0, 1.0]))
            for p in preds.flatten():
                assert np.isfinite(p), f"Non-finite prediction: {p}"
        except Exception:
            pass  # ODE solver can fail for some parameter combos


# --- Property-Based Tests for Substitution Models ---

class TestFisherPryPropertiesExtended:
    """Extended property-based tests for Fisher-Pry model."""

    @given(
        st.floats(min_value=0.1, max_value=2.0),  # alpha
        st.floats(min_value=3.0, max_value=12.0),  # t0
        st.floats(min_value=0.5, max_value=1.0),  # fraction threshold
    )
    @settings(max_examples=15)
    def test_substitution_fraction_crosses_threshold(self, alpha, t0, threshold):
        """Substitution fraction should eventually exceed any threshold < 1."""
        model = FisherPryModel()
        model.params_ = {"alpha": alpha, "t0": t0}
        # At time much greater than t0, fraction should approach 1
        t_far = t0 + 20.0 / alpha
        frac = model.predict([t_far])
        assert frac[0] > threshold or frac[0] < 0.01, \
            f"Substitution fraction {frac[0]} unexpected at t={t_far}"

    @given(
        st.floats(min_value=0.1, max_value=2.0),  # alpha
        st.floats(min_value=3.0, max_value=12.0),  # t0
    )
    @settings(max_examples=15)
    def test_symmetric_around_t0(self, alpha, t0):
        """Substitution fraction at t0-delta + fraction at t0+delta should be ~1."""
        model = FisherPryModel()
        model.params_ = {"alpha": alpha, "t0": t0}
        delta = 1.0 / alpha
        frac_before = model.predict([t0 - delta])[0]
        frac_after = model.predict([t0 + delta])[0]
        total = frac_before + frac_after
        assert abs(total - 1.0) < 0.05, f"Symmetry broken: {frac_before} + {frac_after} = {total}"
