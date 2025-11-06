#!/usr/bin/env python
"""
Mutation testing for the innovate library.
This script runs mutmut to test the quality of our test suite.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_mutation_testing():
    """Run mutation testing on the innovate library."""
    print("Starting mutation testing...")
    
    # Set the environment variable to use our config file
    os.environ['MUTMUT_CONFIG_FILE'] = os.path.join(os.getcwd(), 'mutmut_config.py')
    
    # First, try to run mutation testing on the bass model
    print("Running mutation testing on bass model...")
    try:
        # Use mutmut with our configuration
        cmd = [
            "mutmut",
            "run",
            "--runner",
            "python -m pytest tests/test_property_based_safe.py::test_bass_model_finite_values tests/test_property_based_safe.py::test_bass_model_unfitted_error --tb=no -q"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Successfully completed mutation testing for bass model")
            print(result.stdout)
        else:
            print(f"✗ Error in mutation testing for bass model: {result.stderr}")
        
    except subprocess.TimeoutExpired:
        print("⚠ Timeout during mutation testing for bass model")
    except Exception as e:
        print(f"✗ Exception during mutation testing for bass model: {e}")
    
    # Show mutation testing results
    print("\nMutation testing summary:")
    try:
        result = subprocess.run(["mutmut", "results"], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Could not retrieve mutation results: {e}")
    
    # Show detailed results for survived mutations
    print("\nDetailed results for any surviving mutations:")
    try:
        result = subprocess.run(["mutmut", "show"], capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("No surviving mutations found!")
    except Exception as e:
        print(f"Could not retrieve detailed mutation results: {e}")

if __name__ == "__main__":
    run_mutation_testing()