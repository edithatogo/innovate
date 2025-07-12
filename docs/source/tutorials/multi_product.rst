.. _tutorial_multi_product:

Multi-Product Diffusion Model
=============================

This tutorial demonstrates how to use the `MultiProductDiffusionModel` to simulate the diffusion of multiple competing products.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from innovate.compete.multi_product import MultiProductDiffusionModel

    # Initialize the model for 2 products
    model = MultiProductDiffusionModel(n_products=2)

    # Set the parameters
    model.params_ = {
        "p1": 0.03, "p2": 0.02,
        "q1": 0.1, "q2": 0.15,
        "m1": 1000, "m2": 1200,
        "alpha_1_2": 0.5, "alpha_2_1": 0.3
    }

    # Generate the time points
    t = np.linspace(0, 50, 100)

    # Predict the diffusion
    y = model.predict(t)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, y[:, 0], label='Product 1')
    plt.plot(t, y[:, 1], label='Product 2')
    plt.title("Multi-Product Diffusion Model")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Adopters")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
