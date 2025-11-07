"""Tests for the dynamics base module."""

import pytest
import numpy as np
from abc import ABC, abstractmethod

from innovate.dynamics.base import GrowthCurve, ContagionSpread


class TestGrowthCurve:
    """Test cases for GrowthCurve abstract base class."""
    
    def test_growth_curve_is_abstract(self):
        """Test that GrowthCurve is an abstract class that can't be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GrowthCurve()
    
    def test_growth_curve_subclass_must_implement_compute_growth_rate(self):
        """Test that subclasses must implement compute_growth_rate method."""
        
        class IncompleteGrowthCurve(GrowthCurve):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteGrowthCurve()
        
        class CompleteGrowthCurve(GrowthCurve):
            def compute_growth_rate(self, current_adopters, total_potential, **params):
                return 0.0
        
        # This should work without error
        instance = CompleteGrowthCurve()
        assert instance is not None
        
        # Test the implemented method
        result = instance.compute_growth_rate(10, 100, param1=1.0)
        assert result == 0.0


class TestContagionSpread:
    """Test cases for ContagionSpread abstract base class."""
    
    def test_contagion_spread_is_abstract(self):
        """Test that ContagionSpread is an abstract class that can't be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ContagionSpread()
    
    def test_contagion_spread_subclass_must_implement_differential(self):
        """Test that subclasses must implement differential method."""
        
        class IncompleteContagionSpread(ContagionSpread):
            pass
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteContagionSpread()
        
        class CompleteContagionSpread(ContagionSpread):
            def differential(self, y: np.ndarray, t: float) -> np.ndarray:
                return np.array([0.0])
        
        # This should work without error
        instance = CompleteContagionSpread()
        assert instance is not None
        
        # Test the implemented method
        y = np.array([1.0, 2.0])
        t = 1.0
        result = instance.differential(y, t)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert result[0] == 0.0


def test_abstract_classes_separation():
    """Test that GrowthCurve and ContagionSpread are separate abstract classes."""
    assert GrowthCurve.__name__ == "GrowthCurve"
    assert ContagionSpread.__name__ == "ContagionSpread"
    
    # They should be different classes
    assert GrowthCurve is not ContagionSpread
    
    # They should both be ABCs
    from abc import ABC
    assert issubclass(GrowthCurve, ABC)
    assert issubclass(ContagionSpread, ABC)