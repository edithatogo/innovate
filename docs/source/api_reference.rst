API Reference
=============

This section provides detailed API documentation for all modules in the innovate library.

.. currentmodule:: innovate

Stable public imports
---------------------

The recommended user-facing API is the package-level surface:

.. code-block:: python

   from innovate import BassModel, GompertzModel, LogisticModel, ScipyFitter
   from innovate.compete import MultiProductDiffusionModel, LotkaVolterraModel
   from innovate.substitute import CompositeDiffusionModel, FisherPryModel, NortonBassModel
   from innovate.ecosystem import ComplementaryGoodsModel
   from innovate.backends import use_backend

Legacy deep-module imports remain importable for compatibility, but should not be preferred in new examples.

Core Diffusion Models
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   diffuse.bass
   diffuse.gompertz
   diffuse.logistic

Competition & Market Dynamics  
----------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   compete.competition
   compete.lotka_volterra

Advanced Diffusion Models
-------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   models.advanced
   models.hierarchical
   models.mixture

Technology Substitution
-----------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   substitute.fisher_pry

Hype Cycles & Market Sentiment
------------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   hype.delayed_hype_bass
   hype.hype_cycle
   hype.hype_modified_bass

Innovation Failure Analysis
---------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   fail.analysis

Adoption Categories & User Segments
-----------------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   adopt.categorization

Path Dependence & Lock-in Effects
---------------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   path_dependence.lock_in

Policy Interventions
-------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   policy.intervention

Agent-Based Modeling
--------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   abm.agent
   abm.model
   abm.competitive_diffusion
   abm.disruptive_innovation
   abm.sentiment_hype_cycle

Model Fitting & Parameter Estimation
------------------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   fitters.scipy_fitter
   fitters.bayesian_fitter
   fitters.genetic_fitter

Visualization & Plotting
------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   plots.comparison
   plots.network

Utilities & Preprocessing
-------------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   utils.model_evaluation
   utils.preprocessing
   preprocess.time_series
   preprocess.decomposition

Backend & Base Classes
---------------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   base.base
   backend

Data Handling
-------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   data.market_research
   data.simulated

Causal Inference
---------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst

   causal.counterfactual
