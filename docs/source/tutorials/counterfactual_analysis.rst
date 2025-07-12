.. _tutorial_counterfactual_analysis:

Counterfactual Analysis
=======================

This tutorial demonstrates how to use the `CounterfactualAnalysis` class to conduct "what-if" analysis.

.. code-block:: python

    import numpy as np
    from innovate.diffuse.bass import BassModel
    from innovate.fitters.scipy_fitter import ScipyFitter
    from innovate.causal.counterfactual import CounterfactualAnalysis

    # Generate some synthetic data
    t_data = np.arange(1, 21)
    p, q, m = 0.03, 0.38, 660
    bass_model_true = BassModel()
    bass_model_true.params_ = {'p': p, 'q': q, 'm': m}
    y_data = bass_model_true.predict(t_data)

    # Fit a model to the data
    bass_model = BassModel()
    fitter = ScipyFitter()
    fitter.fit(bass_model, t_data, y_data)

    # Initialize the counterfactual analysis
    analysis = CounterfactualAnalysis(bass_model)

    # Run the baseline forecast
    analysis.run_baseline(t_data)

    # Define a counterfactual scenario
    counterfactual_params = {'p': 0.05} # Increase the coefficient of innovation

    # Run the counterfactual scenario
    analysis.run_counterfactual(
        scenario_name="increased_innovation",
        t=t_data,
        counterfactual_params=counterfactual_params
    )

    # Compare the scenarios
    comparison = analysis.compare_scenarios("increased_innovation")

    # Plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(t_data, comparison['baseline'], label='Baseline')
    plt.plot(t_data, comparison['counterfactual'], label='Counterfactual', linestyle='--')
    plt.title("Counterfactual Analysis")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Adopters")
    plt.legend()
    plt.grid(True)
    plt.show()
