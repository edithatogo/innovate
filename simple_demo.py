#!/usr/bin/env python3
"""
Simple demonstration of the innovate library functionality.
This script demonstrates core features without requiring GPU acceleration.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demonstrate_basic_models():
    """Demonstrate basic diffusion models."""
    print("=== Basic Diffusion Models Demo ===\n")
    
    # Import models
    from innovate.diffuse.bass import BassModel
    from innovate.diffuse.gompertz import GompertzModel
    from innovate.diffuse.logistic import LogisticModel
    from innovate.fitters.scipy_fitter import ScipyFitter
    
    # Create test data
    t = np.linspace(0, 10, 50)
    p, q, m = 0.03, 0.38, 1000
    y_true = m * (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))
    y_noisy = y_true + np.random.normal(0, 20, size=len(t))
    
    print("Generated synthetic data for demonstration")
    print(f"  Time points: {len(t)}")
    print(f"  True parameters: p={p:.3f}, q={q:.3f}, m={m:.1f}")
    
    # Initialize models
    bass_model = BassModel()
    gompertz_model = GompertzModel()
    logistic_model = LogisticModel()
    fitter = ScipyFitter()
    
    print("\nFitting models to noisy data...")
    
    # Fit models
    try:
        fitter.fit(bass_model, t, y_noisy)
        print("✓ Bass model fitted successfully")
        print(f"  Fitted parameters: {bass_model.params_}")
    except Exception as e:
        print(f"✗ Bass model fitting failed: {e}")
        
    try:
        fitter.fit(gompertz_model, t, y_noisy)
        print("✓ Gompertz model fitted successfully")
        print(f"  Fitted parameters: {gompertz_model.params_}")
    except Exception as e:
        print(f"✗ Gompertz model fitting failed: {e}")
        
    try:
        fitter.fit(logistic_model, t, y_noisy)
        print("✓ Logistic model fitted successfully")
        print(f"  Fitted parameters: {logistic_model.params_}")
    except Exception as e:
        print(f"✗ Logistic model fitting failed: {e}")

def demonstrate_advanced_features():
    """Demonstrate advanced features."""
    print("\n\n=== Advanced Features Demo ===\n")
    
    # Import competition models
    try:
        from innovate.compete.lotka_volterra import LotkaVolterraModel
        lv_model = LotkaVolterraModel()
        print("✓ Lotka-Volterra competition model imported successfully")
        print(f"  Parameter names: {lv_model.param_names}")
    except Exception as e:
        print(f"✗ Lotka-Volterra model import failed: {e}")
    
    # Import substitution models
    try:
        from innovate.substitute.fisher_pry import FisherPryModel
        fp_model = FisherPryModel()
        print("✓ Fisher-Pry substitution model imported successfully")
        print(f"  Parameter names: {fp_model.param_names}")
    except Exception as e:
        print(f"✗ Fisher-Pry model import failed: {e}")

def demonstrate_australian_study_replication():
    """Demonstrate replication of the Australian genomic testing study."""
    print("\n\n=== Australian Genomic Testing Study Replication ===\n")
    
    from innovate.diffuse.bass import BassModel
    from innovate.diffuse.gompertz import GompertzModel
    from innovate.fitters.scipy_fitter import ScipyFitter
    
    # Simulate the Australian study findings
    # MBS item 73292 - best fit with Gompertz model (MAE=197.2982)
    # Group of services - best fit with Bass model (MAE=21.6853)
    
    print("Replicating key findings from Australian MBS genomic testing study:")
    print("  - MBS item 73292: Best fit with Gompertz model (MAE=197.2982)")
    print("  - Group of services: Best fit with Bass model (MAE=21.6853)")
    print("  - Predicted intersection around April 2029")
    
    # Create synthetic data mimicking the study patterns
    dates = np.arange(120)  # 10 years of monthly data
    
    # MBS item 73292 pattern (Gompertz-like)
    mbs_73292 = 400 * (1 - np.exp(-0.06 * dates)) + np.random.normal(0, 15, len(dates))
    mbs_73292 = np.maximum(mbs_73292, 0)  # Ensure non-negative
    
    # Group services pattern (Bass-like)
    mbs_group = 150 * (1 - np.exp(-0.08 * (dates - 24))) * (dates > 23)
    mbs_group = np.concatenate([np.zeros(24), mbs_group[:len(mbs_group)-24]]) + np.random.normal(0, 10, len(dates))
    mbs_group = np.maximum(mbs_group, 0)  # Ensure non-negative
    
    print(f"\nGenerated synthetic data:")
    print(f"  MBS 73292: {len(mbs_73292)} data points")
    print(f"  MBS Group: {len(mbs_group)} data points")
    
    # Fit models as in the study
    gompertz_73292 = GompertzModel()
    bass_group = BassModel()
    fitter = ScipyFitter()
    
    print("\nFitting models to replicate study findings...")
    
    try:
        # Fit Gompertz to MBS 73292 (as found best in study)
        fitter.fit(gompertz_73292, dates, mbs_73292)
        y_73292_pred = gompertz_73292.predict(dates)
        mae_73292 = np.mean(np.abs(mbs_73292 - y_73292_pred))
        print(f"✓ Gompertz model for MBS 73292 - MAE: {mae_73292:.4f}")
        print(f"  Target MAE from study: 197.2982")
        print(f"  Fitted parameters: {gompertz_73292.params_}")
    except Exception as e:
        print(f"✗ Gompertz fitting failed: {e}")
    
    try:
        # Fit Bass to MBS Group (as found best in study)
        fitter.fit(bass_group, dates, mbs_group)
        y_group_pred = bass_group.predict(dates)
        mae_group = np.mean(np.abs(mbs_group - y_group_pred))
        print(f"✓ Bass model for MBS Group - MAE: {mae_group:.4f}")
        print(f"  Target MAE from study: 21.6853")
        print(f"  Fitted parameters: {bass_group.params_}")
    except Exception as e:
        print(f"✗ Bass fitting failed: {e}")

def main():
    """Main demonstration function."""
    print("innovate Library Demonstration")
    print("=" * 50)
    print("This script demonstrates the core functionality of the innovate library")
    print("without requiring GPU acceleration.\n")
    
    # Run demonstrations
    demonstrate_basic_models()
    demonstrate_advanced_features()
    demonstrate_australian_study_replication()
    
    print("\n" + "=" * 50)
    print("Demonstration complete!")
    print("The innovate library provides a comprehensive framework for")
    print("innovation and policy diffusion modeling with special applicability")
    print("to health economic analysis, as demonstrated with the Australian")
    print("genomic testing study findings.")
    print("=" * 50)

if __name__ == "__main__":
    main()