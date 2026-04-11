"""
Example: Basic Usage of the Innovate Library
============================================

This example demonstrates the basic usage of the innovate library for modeling
innovation and policy diffusion using the Bass model.
"""

import matplotlib.pyplot as plt
import numpy as np

from innovate.diffuse.bass import BassModel
from innovate.fitters.scipy_fitter import ScipyFitter


def main():
    print("Innovate Library: Basic Usage Example")
    print("=" * 40)

    # Generate synthetic data using known parameters
    print("\n1. Generating synthetic data...")
    true_p = 0.03  # coefficient of innovation
    true_q = 0.35  # coefficient of imitation
    true_m = 1000  # market potential

    # Generate time points
    t_data = np.linspace(0, 10, 50)

    # Generate cumulative adoption using the Bass model equation
    cumulative_adoption = (
        true_m
        * (1 - np.exp(-(true_p + true_q) * t_data))
        / (1 + (true_q / true_p) * np.exp(-(true_p + true_q) * t_data))
    )

    # Add some noise to make it more realistic
    noisy_adoption = cumulative_adoption + np.random.normal(0, 20, size=cumulative_adoption.shape)
    noisy_adoption = np.maximum(noisy_adoption, 0)  # Ensure non-negative values

    print(f"Generated {len(t_data)} data points")
    print(f"Adoption ranges from {noisy_adoption.min():.1f} to {noisy_adoption.max():.1f}")

    # Create model and fitter
    print("\n2. Creating model and fitting to data...")
    model = BassModel()
    fitter = ScipyFitter()

    # Fit the model to the data
    fitted_model = model.fit(fitter, t_data, noisy_adoption)

    # Print the fitted parameters
    print("\nFitted Parameters:")
    for param_name, param_value in fitted_model.params_.items():
        print(f"  {param_name}: {param_value:.4f}")

    # Get the R² score
    r_squared = fitted_model.score(t_data, noisy_adoption)
    print(f"\nR² Score: {r_squared:.4f}")

    # Generate predictions
    print("\n3. Generating predictions...")
    t_pred = np.linspace(0, 12, 100)
    predictions = fitted_model.predict(t_pred)

    # Plot results
    print("\n4. Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(t_data, noisy_adoption, "bo", label="Observed Data", markersize=6)
    plt.plot(t_pred, predictions, "r-", label="Bass Model Fit", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Adoption")
    plt.title("Bass Model: Fitting Innovation Diffusion Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Show adoption rate prediction
    print("\n5. Predicting adoption rate...")
    adoption_rates = fitted_model.predict_adoption_rate(t_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(t_pred, adoption_rates, "g-", label="Adoption Rate", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Adoption Rate")
    plt.title("Rate of New Adoptions Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nExample completed successfully!")
    print(f"Model explains {r_squared:.1%} of the variance in the data.")


if __name__ == "__main__":
    main()
