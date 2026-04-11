"""Example of a multi-product diffusion model with a failing product."""
import numpy as np

from innovate.compete.competition import MultiProductDiffusionModel


def run_failed_adoption_example() -> None:
    """Run a simplified simulation of product adoption to identify failed products."""
    # Define parameters for a multi-product model where one product is
    # designed to fail
    # Product A: Moderate success
    # Product B: Low adoption, likely to fail
    # Product C: High success
    p_vals = [0.03, 0.005, 0.04]  # Intrinsic adoption rates
    q_matrix = [
        [
            0.3,
            0.05,
            0.02,
        ],  # Q[0,0] = imitation for ProdA from ProdA, Q[0,1] = ProdA from ProdB, etc.
        [
            0.01,
            0.1,
            0.01,
        ],  # ProdB has low internal imitation and low influence from others
        [0.05, 0.02, 0.4],  # ProdC has high internal imitation
    ]
    m_vals = [1000, 200, 1200]  # Ultimate market potentials
    product_names = ["Product A", "Product B", "Product C"]

    model = MultiProductDiffusionModel(
        p=p_vals,
        Q=q_matrix,
        m=m_vals,
        names=product_names,
    )

    time_horizon = np.arange(1, 51)  # 50 time points
    model.predict(time_horizon)


if __name__ == "__main__":
    run_failed_adoption_example()
