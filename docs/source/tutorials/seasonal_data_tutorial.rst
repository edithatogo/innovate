.. _seasonal_data_tutorial:

===========================================
Handling Seasonal Data with Diffusion Models
===========================================

Real-world data often exhibits seasonality, which can violate the assumptions of simple diffusion models. This tutorial demonstrates how to use the ``innovate`` library in combination with ``statsmodels`` to handle seasonal data by decomposing a time series into its trend, seasonal, and residual components.

Introduction to Seasonal Decomposition
--------------------------------------

Seasonal decomposition is a statistical technique that separates a time series into three components:

- **Trend**: The underlying long-term movement in the data.
- **Seasonality**: The repeating short-term cycles in the data.
- **Residuals**: The random, irregular fluctuations in the data.

By decomposing the time series, we can isolate the trend component and fit our diffusion model to it. This allows the model to capture the underlying diffusion process without being confounded by seasonality.

A Simple Example: Fitting a Logistic Model to a Seasonal Time Series
---------------------------------------------------------------------

Let's walk through an example of fitting a ``LogisticModel`` to some synthetic seasonal data.

1. Generate Seasonal Data
~~~~~~~~~~~~~~~~~~~~~~~~~

First, we'll generate some synthetic data with a clear seasonal pattern.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import seasonal_decompose
    from innovate.diffuse import LogisticModel
    from innovate.fitters import BayesianFitter

    # Generate a time series with a seasonal component
    t_seasonal = np.linspace(0, 100, 500)
    seasonal_component = 10 * np.sin(2 * np.pi * t_seasonal / 25)
    trend_component = 0.1 * t_seasonal**2
    y_seasonal = trend_component + seasonal_component + np.random.normal(0, 5, len(t_seasonal))

    plt.figure(figsize=(10, 6))
    plt.plot(t_seasonal, y_seasonal, label='Seasonal Data')
    plt.legend()
    plt.show()

2. Decompose the Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the ``seasonal_decompose`` function from ``statsmodels`` to separate the trend from the seasonal component.

.. code-block:: python

    # Decompose the time series
    decomposition = seasonal_decompose(y_seasonal, model='additive', period=25)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax1.plot(y_seasonal, label='Original')
    ax1.legend()
    ax2.plot(trend, label='Trend')
    ax2.legend()
    ax3.plot(seasonal, label='Seasonal')
    ax3.legend()
    ax4.plot(residual, label='Residual')
    ax4.legend()
    plt.show()

3. Fit the Model to the Trend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we can fit our diffusion model to the extracted trend component. This allows the model to capture the underlying diffusion process without being confounded by seasonality.

.. code-block:: python

    # Fit the model to the trend component
    # Note: We need to remove the NaNs from the trend component
    
    t_trend = t_seasonal[~np.isnan(trend)]
    y_trend = trend[~np.isnan(trend)]
    
    model_for_trend = LogisticModel()
    fitter_for_trend = BayesianFitter(model=model_for_trend, draws=2000, tune=1000, chains=4)
    fitter_for_trend.fit(t_trend, y_trend)

    # Get the parameter estimates for the trend
    trend_params = fitter_for_trend.get_parameter_estimates()
    print("Parameter Estimates for the Trend:")
    print(trend_params)

Conclusion
----------

By using STL decomposition, you can extend the applicability of diffusion models to seasonal data, allowing you to model the underlying growth dynamics even in the presence of periodic fluctuations. This is a powerful technique for improving the accuracy of your diffusion models when dealing with real-world data.
