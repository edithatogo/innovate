"""Tests for adopt/categorization module."""

import numpy as np
import pytest

from innovate.backend import use_backend

use_backend("numpy")

from innovate.adopt.categorization import categorize_adopters
from innovate.diffuse.bass import BassModel


class TestCategorizeAdopters:
    """Test adopter categorization function."""

    def _make_fitted_model(self):
        """Create a fitted Bass model for testing."""
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 100.0}
        return model

    def test_basic_categorization(self):
        """Test basic adopter categorization."""
        model = self._make_fitted_model()
        t = np.linspace(0, 15, 100)
        result = categorize_adopters(model, t)
        assert len(result) == 100
        assert "time" in result.columns
        assert "adoption_rate" in result.columns
        assert "category" in result.columns

    def test_all_categories_present(self):
        """Test that all 5 categories appear in results."""
        model = self._make_fitted_model()
        t = np.linspace(0, 20, 200)
        result = categorize_adopters(model, t)
        categories = set(result["category"].unique())
        assert len(categories) >= 3  # At least 3 categories should appear

    def test_innovators_at_early_times(self):
        """Test that early times are classified as Innovators."""
        model = self._make_fitted_model()
        t = np.linspace(0, 15, 100)
        result = categorize_adopters(model, t)
        # First few times should be Innovators
        early_cats = result.head(5)["category"].unique()
        assert len(early_cats) >= 1

    def test_laggards_at_late_times(self):
        """Test that late times are classified as Laggards."""
        model = self._make_fitted_model()
        t = np.linspace(0, 20, 200)
        result = categorize_adopters(model, t)
        # Last few times should be Laggards
        late_cats = result.tail(5)["category"].unique()
        assert "Laggards" in late_cats

    def test_adoption_rate_positive(self):
        """Test that adoption rates are positive."""
        model = self._make_fitted_model()
        t = np.linspace(0, 15, 100)
        result = categorize_adopters(model, t)
        assert all(result["adoption_rate"] >= 0)

    def test_times_preserved(self):
        """Test that input times are preserved in output."""
        model = self._make_fitted_model()
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = categorize_adopters(model, t)
        assert list(result["time"]) == list(t)

    def test_single_time_point(self):
        """Test categorization with single time point."""
        model = self._make_fitted_model()
        t = np.array([5.0])
        result = categorize_adopters(model, t)
        assert len(result) == 1
        assert result["category"].iloc[0] in [
            "Innovators",
            "Early Adopters",
            "Early Majority",
            "Late Majority",
            "Laggards",
        ]

    def test_unfitted_model_raises(self):
        """Test that unfitted model raises error."""
        model = BassModel()  # Not fitted
        t = np.array([1.0, 2.0, 3.0])
        with pytest.raises(RuntimeError):
            categorize_adopters(model, t)
