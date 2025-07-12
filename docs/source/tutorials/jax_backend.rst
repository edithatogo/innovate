.. _tutorial_jax_backend:

JAX Backend for High-Performance Computing
==========================================

This tutorial demonstrates how to use the JAX backend for high-performance computing.

.. code-block:: python

    from innovate.backend import use_backend

    # Switch to the JAX backend
    use_backend("jax")

    # Now, all models will use the JAX backend for their computations.
    # For example, let's use the BassModel:

    import numpy as np
    import matplotlib.pyplot as plt
    from innovate.diffuse.bass import BassModel

    # Initialize the model
    model = BassModel()

    # Set the parameters
    model.params_ = {
        "p": 0.03, "q": 0.38, "m": 1000
    }

    # Generate the time points
    t = np.linspace(0, 20, 100)

    # Predict the diffusion
    y = model.predict(t)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label=f'p=0.03, q=0.38, m=1000')
    plt.title("Bass Diffusion Model with JAX Backend")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Adopters")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
