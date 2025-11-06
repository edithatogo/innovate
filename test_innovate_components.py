#!/usr/bin/env python3
"""
Test script to verify the innovate library components can be imported and instantiated.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test importing key components of the innovate library."""
    print("Testing imports of innovate library components...")
    
    try:
        # Test importing basic models
        from innovate.diffuse.bass import BassModel
        from innovate.diffuse.gompertz import GompertzModel
        from innovate.diffuse.logistic import LogisticModel
        
        print("✓ Successfully imported basic diffusion models")
        
        # Test importing competition models
        from innovate.compete.lotka_volterra import LotkaVolterraModel
        print("✓ Successfully imported competition models")
        
        # Test importing substitution models
        from innovate.substitute.fisher_pry import FisherPryModel
        print("✓ Successfully imported substitution models")
        
        # Test importing fitters
        from innovate.fitters.scipy_fitter import ScipyFitter
        print("✓ Successfully imported fitters")
        
        # Test instantiating models
        bass_model = BassModel()
        gompertz_model = GompertzModel()
        logistic_model = LogisticModel()
        lotka_volterra_model = LotkaVolterraModel()
        fisher_pry_model = FisherPryModel()
        fitter = ScipyFitter()
        
        print("✓ Successfully instantiated all models and fitters")
        
        # Test accessing basic properties
        print(f"  Bass model param names: {bass_model.param_names}")
        print(f"  Gompertz model param names: {gompertz_model.param_names}")
        print(f"  Logistic model param names: {logistic_model.param_names}")
        print(f"  Lotka-Volterra model param names: {lotka_volterra_model.param_names}")
        print(f"  Fisher-Pry model param names: {fisher_pry_model.param_names}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_backend_setup():
    """Test backend setup."""
    print("\nTesting backend setup...")
    
    try:
        # Test importing backend
        from innovate import backend
        print("✓ Successfully imported backend module")
        
        # Test checking current backend
        current_backend = getattr(backend, 'current_backend', None)
        if current_backend:
            print(f"  Current backend: {type(current_backend).__name__}")
        else:
            print("  No current backend set")
            
        return True
    except Exception as e:
        print(f"✗ Backend setup error: {e}")
        return False

def main():
    """Main test function."""
    print("=== innovate Library Component Test ===\n")
    
    success = True
    
    # Test imports
    success &= test_imports()
    
    # Test backend
    success &= test_backend_setup()
    
    print("\n" + "="*50)
    if success:
        print("✓ All tests passed! The innovate library components are working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*50)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())