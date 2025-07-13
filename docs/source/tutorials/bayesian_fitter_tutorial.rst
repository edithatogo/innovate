.. _bayesian_fitter_tutorial:

==================================
Bayesian Fitting for Diffusion Models
==================================

This tutorial provides a guide to using the ``BayesianFitter`` in the ``innovate`` library. Bayesian methods offer a powerful way to estimate the parameters of diffusion models, providing not just point estimates but entire posterior distributions that quantify uncertainty.

Introduction to Bayesian Fitting
--------------------------------

Traditional fitting methods, like those based on least squares, provide a single "best" estimate for model parameters. In contrast, Bayesian inference treats parameters as random variables and seeks to determine their probability distribution based on the observed data. This approach has several key advantages:

- **Uncertainty Quantification**: It provides a full posterior distribution for each parameter, allowing us to quantify our uncertainty. We can compute credible intervals (the Bayesian equivalent of confidence intervals) to understand the range of plausible parameter values.
- **Regularization**: Priors naturally regularize the model, preventing overfitting and leading to more stable estimates, especially with noisy or sparse data.
- **Flexibility**: The Bayesian framework is highly flexible, allowing for the incorporation of prior knowledge and the construction of complex hierarchical models.

The ``innovate`` library's ``BayesianFitter`` uses the powerful ``PyMC`` library under the hood to perform Markov Chain Monte Carlo (MCMC) sampling.

A Simple Example: Fitting a Logistic Model
------------------------------------------

Let's walk through an example of fitting a ``LogisticModel`` to some synthetic data.

1. Generate Synthetic Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we'll create a ``LogisticModel``, set its parameters, and generate some noisy data.

.. code-block:: python

    import numpy as np
    from innovate.diffuse import LogisticModel
    from innovate.fitters import BayesianFitter

    # 1. Define the true model and generate data
    true_model = LogisticModel()
    true_model.set_params(L=1000, k=0.1, x0=50)

    t = np.linspace(0, 100, 50)
    true_adoptions = true_model.predict(t)
    noise = np.random.normal(0, 30, len(t))
    y = true_adoptions + noise
    y[y < 0] = 0 # Ensure non-negative adoptions

2. Fit the Model with ``BayesianFitter``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we create an instance of the ``LogisticModel`` to be fitted and an instance of the ``BayesianFitter``. Then, we call the ``fit`` method.

.. code-block:: python

    # 2. Create a new model instance and the fitter
    model_to_fit = LogisticModel()
    fitter = BayesianFitter(model=model_to_fit, draws=2000, tune=1000, chains=4)

    # 3. Fit the model to the data
    fitter.fit(t, y)

    print("Fitting complete.")

3. Interpreting the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The real power of the Bayesian approach lies in the rich information we get from the posterior distribution.

Parameter Estimates
^^^^^^^^^^^^^^^^^^^

We can get the mean of the posterior distribution for each parameter, which serves as our point estimate.

.. code-block:: python

    # Get the mean of the posterior as parameter estimates
    parameter_estimates = fitter.get_parameter_estimates()
    print("Parameter Estimates (Posterior Mean):")
    print(parameter_estimates)

Credible Intervals
^^^^^^^^^^^^^^^^^^

We can also compute credible intervals to understand the uncertainty in our estimates. For example, a 95% credible interval means that there is a 95% probability that the true parameter value lies within the interval.

.. code-block:: python

    # Get 95% credible intervals
    credible_intervals = fitter.get_confidence_intervals(alpha=0.05)
    print("\n95% Credible Intervals:")
    print(credible_intervals)

Visualizing the Posterior
^^^^^^^^^^^^^^^^^^^^^^^^^

For a deeper understanding, we can use libraries like ``ArviZ`` to plot the posterior distributions and diagnostic plots. The ``fitter.trace`` object is a ``pymc.backends.base.MultiTrace`` object that can be used with ``ArviZ``.

.. code-block:: python

    import arviz as az

    # Plot the posterior distributions
    az.plot_posterior(fitter.trace)

Full Summary Statistics
^^^^^^^^^^^^^^^^^^^^^^^

The ``get_summary`` method provides a comprehensive summary of the posterior, including the mean, standard deviation, credible intervals, and diagnostic statistics like ``r_hat`` (which should be close to 1.0 to indicate convergence).

.. code-block:: python

    # Get a full summary of the posterior
    summary = fitter.get_summary()
    print("\nFull Posterior Summary:")
    print(summary)


Conclusion
----------

The ``BayesianFitter`` provides a robust and powerful alternative for fitting diffusion models in the ``innovate`` library. By leveraging Bayesian inference, you can gain deeper insights into parameter uncertainty, leading to more reliable and informative models. This is especially valuable when dealing with the noisy, real-world data often encountered in innovation diffusion studies.
