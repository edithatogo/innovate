innovate.models package
=======================

Advanced diffusion workflows for uncertainty-aware adoption analysis.

The ``innovate.models`` namespace exposes the canonical advanced model
families added by the advanced diffusion inference track:

- ``AdvancedDiffusionModel`` and ``AdvancedModelSummary`` for a shared
  advanced-model contract
- ``HierarchicalModel`` for grouped or partially pooled diffusion analysis
- ``MixtureModel`` for latent-class diffusion segmentation
- ``LatentProcessDiffusionModel`` for state-space style residual dynamics
- ``RegimeSwitchingDiffusionModel`` for changepoint-aware structural breaks

Backend requirements
--------------------

- The advanced wrappers are designed to run on the NumPy backend.
- The regime-switching workflow uses ``ruptures`` when available. If the
  dependency is missing, only the changepoint-aware workflow is unavailable.
- These models are intended for empirical research settings where uncertainty
  summaries, simulations, or structural-break analysis matter more than raw
  throughput.

Submodules
----------

.. toctree::
   :maxdepth: 4

   innovate.models.advanced
   innovate.models.hierarchical
   innovate.models.mixture

Module contents
---------------

.. automodule:: innovate.models
   :members:
   :show-inheritance:
   :undoc-members:
