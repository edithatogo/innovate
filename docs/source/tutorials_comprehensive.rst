Tutorials & Examples
====================

Welcome to the comprehensive tutorials and examples for the innovate library. These tutorials cover everything from basic usage to advanced modeling techniques.

.. note::
   All Jupyter notebooks referenced here are available in the `examples/ <https://github.com/doughnutsz/innovate/tree/main/examples>`_ directory of the repository.

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   ../../../examples/basic_diffusion_modeling
   ../../../examples/model_comparison_framework

Core Diffusion Models
---------------------

Learn how to use the fundamental diffusion models for innovation adoption analysis.

.. toctree::
   :maxdepth: 2

   ../../../examples/bass_competition_tutorial
   ../../../examples/model_selection_guide

Advanced Modeling Techniques
----------------------------

Explore sophisticated modeling approaches for complex innovation scenarios.

.. toctree::
   :maxdepth: 2

   ../../../examples/agent_based_modeling_case_study
   ../../../examples/policy_analysis_case_studies

Data Preparation & Analysis
---------------------------

Master data preprocessing and preparation techniques for diffusion modeling.

.. toctree::
   :maxdepth: 2

   ../../../examples/time_series_preprocessing_tutorial
   ../../../examples/data_preprocessing_guide

Real-World Applications
----------------------

See how to apply innovation diffusion models to real-world scenarios and datasets.

.. toctree::
   :maxdepth: 2

   ../../../examples/case_study_renewable_energy
   ../../../examples/case_study_social_media

Interactive Examples
-------------------

Hands-on examples you can run and modify to learn the library.

Basic Usage Examples
~~~~~~~~~~~~~~~~~~~

* **Bass Model Fitting**: Learn to fit the Bass diffusion model to adoption data
* **Competition Analysis**: Model competitive dynamics between innovations  
* **Parameter Estimation**: Explore different fitting techniques and their trade-offs

Advanced Applications
~~~~~~~~~~~~~~~~~~~~

* **Policy Impact Analysis**: Assess how policy interventions affect innovation adoption
* **Agent-Based Modeling**: Build complex models with heterogeneous agents
* **Time Series Analysis**: Preprocess and analyze longitudinal adoption data

Case Studies
~~~~~~~~~~~~

* **Renewable Energy Adoption**: Model solar panel diffusion with policy scenarios
* **Electric Vehicle Growth**: Analyze EV adoption with infrastructure effects
* **Digital Health Technology**: Study telemedicine adoption during healthcare crises

Quick Reference
---------------

Key Concepts
~~~~~~~~~~~~

* **Diffusion Models**: Bass, Gompertz, Logistic models for adoption curves
* **Competition Dynamics**: Multi-product models with competitive effects
* **Policy Interventions**: Modeling the impact of subsidies and regulations
* **Agent-Based Models**: Individual-level modeling with emergent adoption patterns

Model Selection Guide
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Selection Quick Reference
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model Type
     - Best For
     - Key Parameters
     - Use Cases
   * - Bass Model
     - Standard S-curve adoption
     - p (innovation), q (imitation), m (market size)
     - New product launches, technology adoption
   * - Gompertz Model  
     - Slow initial growth
     - a (ceiling), b (displacement), c (growth rate)
     - Infrastructure, platform adoption
   * - Logistic Model
     - Symmetric growth
     - L (carrying capacity), k (growth rate), x0 (midpoint)
     - Population models, resource-limited growth
   * - Multi-Product
     - Competing innovations
     - Multiple p, Q matrix, m vector
     - Market competition, substitutes

Performance Tips
~~~~~~~~~~~~~~~~

* **Data Quality**: Ensure time series data is clean and properly indexed
* **Model Selection**: Use cross-validation for robust model comparison
* **Parameter Bounds**: Set reasonable bounds during fitting to avoid overfitting
* **Backend Choice**: Use JAX backend for large-scale computations

Troubleshooting
~~~~~~~~~~~~~~~

Common Issues and Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Fitting Convergence**: Try different initial parameter values or fitting algorithms
* **Negative Predictions**: Check for data quality issues or inappropriate model choice
* **Poor Fit Quality**: Consider data transformations or alternative model specifications
* **Performance Issues**: Switch to JAX backend for computational efficiency

Getting Help
~~~~~~~~~~~~

* **GitHub Issues**: Report bugs or request features at `GitHub Issues <https://github.com/doughnutsz/innovate/issues>`_
* **Documentation**: Full API reference available in the :doc:`api_reference`
* **Examples**: All notebook examples are in the `examples/ directory <https://github.com/doughnutsz/innovate/tree/main/examples>`_