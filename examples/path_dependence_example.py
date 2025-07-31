import numpy as np
import matplotlib.pyplot as plt
from innovate.path_dependence.lock_in import LockInModel


def run_path_dependence_example():
    print("--- Running Path Dependence Example ---")

    # Define parameters to illustrate path dependence
    # Tech 1 has a slight initial advantage or stronger network effects
    params = {
        "alpha1": 0.1,
        "alpha2": 0.1,  # Intrinsic growth rates
        "beta1": 0.02,
        "beta2": 0.01,  # Network effect strengths (Tech 1 stronger)
        "gamma1": 0.005,
        "gamma2": 0.005,  # Negative influence of competitor
        "m": 1000.0,  # Total market potential
    }

    model = LockInModel()
    model.params_ = params

    t = np.arange(0, 100, 1)  # Time horizon
    y0_scenario1 = [10.0, 1.0]  # Scenario 1: Tech 1 starts with an advantage
    y0_scenario2 = [1.0, 10.0]  # Scenario 2: Tech 2 starts with an advantage
    y0_scenario3 = [5.0, 5.0]  # Scenario 3: Equal start

    # Predict for Scenario 1
    predictions_s1 = model.predict(t, y0_scenario1)
    # Predict for Scenario 2
    predictions_s2 = model.predict(t, y0_scenario2)
    # Predict for Scenario 3
    predictions_s3 = model.predict(t, y0_scenario3)

    # Plotting the results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Lock-in Model: Path Dependence in Competing Technologies")

    # Scenario 1 Plot
    axes[0].plot(t, predictions_s1[:, 0], label="Technology 1", color="blue")
    axes[0].plot(
        t, predictions_s1[:, 1], label="Technology 2", color="red", linestyle="--"
    )
    axes[0].set_title(
        f"Scenario 1: Initial (Tech 1={y0_scenario1[0]}, Tech 2={y0_scenario1[1]})"
    )
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Cumulative Adoptions")
    axes[0].legend()
    axes[0].grid(True)

    # Scenario 2 Plot
    axes[1].plot(t, predictions_s2[:, 0], label="Technology 1", color="blue")
    axes[1].plot(
        t, predictions_s2[:, 1], label="Technology 2", color="red", linestyle="--"
    )
    axes[1].set_title(
        f"Scenario 2: Initial (Tech 1={y0_scenario2[0]}, Tech 2={y0_scenario2[1]})"
    )
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True)

    # Scenario 3 Plot
    axes[2].plot(t, predictions_s3[:, 0], label="Technology 1", color="blue")
    axes[2].plot(
        t, predictions_s3[:, 1], label="Technology 2", color="red", linestyle="--"
    )
    axes[2].set_title(
        f"Scenario 3: Initial (Tech 1={y0_scenario3[0]}, Tech 2={y0_scenario3[1]})"
    )
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_path_dependence_example()
