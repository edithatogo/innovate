"""Input validation tests for the innovate library."""
import numpy as np
import pytest
from innovate.backend import use_backend

# Use numpy backend to avoid JAX-related issues
use_backend('numpy')

from innovate.diffuse.bass import BassModel


class TestInputValidation:
    """Test input validation for the BassModel."""
    
    def test_bass_model_initialization_validation(self):
        """Test validation in BassModel initialization."""
        # Valid initialization
        model = BassModel(covariates=["advertising"])
        assert model.covariates == ["advertising"]
        
        # Valid initialization with t_event
        model = BassModel(t_event=5.0)
        assert model.t_event == 5.0
        
        # Valid initialization with no covariates
        model = BassModel()
        assert model.covariates == []
        
        # Invalid t_event type
        with pytest.raises(TypeError):
            BassModel(t_event="invalid")
        
        # Invalid covariates type (not a sequence)
        with pytest.raises(TypeError):
            BassModel(covariates="invalid")
        
        # Invalid covariate element type
        with pytest.raises(TypeError):
            BassModel(covariates=[1, 2, 3])
    
    def test_initial_guesses_validation(self):
        """Test validation in initial_guesses method."""
        model = BassModel()
        
        # Valid inputs
        t = [0, 1, 2, 3]
        y = [10, 20, 30, 40]
        guesses = model.initial_guesses(t, y)
        assert isinstance(guesses, dict)
        assert "p" in guesses
        
        # Invalid t type
        with pytest.raises(TypeError):
            model.initial_guesses("invalid", y)
        
        # Invalid y type
        with pytest.raises(TypeError):
            model.initial_guesses(t, "invalid")
        
        # Empty sequences
        with pytest.raises(ValueError):
            model.initial_guesses([], y)
        
        with pytest.raises(ValueError):
            model.initial_guesses(t, [])
        
        # Non-numeric sequences
        with pytest.raises(TypeError):
            model.initial_guesses(["a", "b"], y)
        
        # Different length sequences
        with pytest.raises(ValueError):
            model.initial_guesses([0, 1], [10, 20, 30])
    
    def test_bounds_validation(self):
        """Test validation in bounds method."""
        model = BassModel()
        
        # Valid inputs
        t = [0, 1, 2, 3]
        y = [10, 20, 30, 40]
        bounds = model.bounds(t, y)
        assert isinstance(bounds, dict)
        assert "p" in bounds
        
        # Invalid inputs - same validation as initial_guesses
        with pytest.raises(TypeError):
            model.bounds("invalid", y)
        
        with pytest.raises(TypeError):
            model.bounds(t, "invalid")
    
    def test_predict_validation(self):
        """Test validation in predict method."""
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        
        # Valid input
        t = [0, 1, 2, 3]
        result = model.predict(t)
        assert len(result) == len(t)
        
        # Invalid t type
        with pytest.raises(TypeError):
            model.predict("invalid")
        
        # Empty t
        with pytest.raises(ValueError):
            model.predict([])
        
        # Negative time values
        with pytest.raises(ValueError):
            model.predict([-1, 0, 1])
        
        # Invalid covariates
        with pytest.raises(ValueError):
            model.predict(t, {"invalid_covariate": [1, 2, 3, 4]})
        
        # Covariates with wrong length
        with pytest.raises(ValueError):
            model.predict(t, {"invalid_covariate": [1, 2]})
        
        # Unfitted model
        unfitted_model = BassModel()
        with pytest.raises(RuntimeError):
            unfitted_model.predict(t)
    
    def test_score_validation(self):
        """Test validation in score method."""
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        
        # Valid inputs
        t = [0, 1, 2, 3]
        y = [10, 20, 30, 40]
        score = model.score(t, y)
        assert isinstance(score, (int, float))
        
        # Invalid t type
        with pytest.raises(TypeError):
            model.score("invalid", y)
        
        # Invalid y type
        with pytest.raises(TypeError):
            model.score(t, "invalid")
        
        # Different length sequences
        with pytest.raises(ValueError):
            model.score([0, 1], [10, 20, 30])
        
        # Negative values in y
        with pytest.raises(ValueError):
            model.score(t, [-1, 10, 20, 30])
        
        # Invalid covariates
        with pytest.raises(ValueError):
            model.score(t, y, {"invalid_covariate": [1, 2, 3, 4]})
    
    def test_predict_adoption_rate_validation(self):
        """Test validation in predict_adoption_rate method."""
        model = BassModel()
        model.params_ = {"p": 0.03, "q": 0.38, "m": 1000}
        
        # Valid input
        t = [0, 1, 2, 3]
        result = model.predict_adoption_rate(t)
        assert len(result) == len(t)
        
        # Invalid t type
        with pytest.raises(TypeError):
            model.predict_adoption_rate("invalid")
        
        # Empty t
        with pytest.raises(ValueError):
            model.predict_adoption_rate([])
        
        # Negative time values
        with pytest.raises(ValueError):
            model.predict_adoption_rate([-1, 0, 1])
        
        # Unfitted model
        unfitted_model = BassModel()
        with pytest.raises(RuntimeError):
            unfitted_model.predict_adoption_rate(t)


def test_numeric_sequence_validation():
    """Test the validation utilities directly."""
    from innovate.utils.validation import validate_sequence_numeric
    
    # Valid sequences
    result = validate_sequence_numeric([1, 2, 3], "test")
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    
    result = validate_sequence_numeric((1.0, 2.0, 3.0), "test")
    assert isinstance(result, np.ndarray)
    
    # Invalid sequences
    with pytest.raises(TypeError):
        validate_sequence_numeric("not a sequence", "test")
    
    with pytest.raises(TypeError):
        validate_sequence_numeric([1, "invalid", 3], "test")
    
    with pytest.raises(ValueError):
        validate_sequence_numeric(None, "test")


if __name__ == "__main__":
    test_instance = TestInputValidation()
    test_instance.test_bass_model_initialization_validation()
    print("✓ Initialization validation test passed")
    
    test_instance.test_initial_guesses_validation()
    print("✓ Initial guesses validation test passed")
    
    test_instance.test_bounds_validation()
    print("✓ Bounds validation test passed")
    
    test_instance.test_predict_validation()
    print("✓ Predict validation test passed")
    
    test_instance.test_score_validation()
    print("✓ Score validation test passed")
    
    test_instance.test_predict_adoption_rate_validation()
    print("✓ Predict adoption rate validation test passed")
    
    test_numeric_sequence_validation()
    print("✓ Numeric sequence validation test passed")
    
    print("All input validation tests passed!")