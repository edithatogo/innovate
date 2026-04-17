innovate.benchmarks package
===========================

Benchmark corpus, model-card, and suite helpers for reproducible evaluation.

The ``innovate.benchmarks`` namespace exposes the canonical benchmark corpus,
model-card registry, and stable suite runner used by the Conductor workflow.
It is the right place to discover:

- stable benchmark cases and their identifiers
- synchronized model cards for stable model families
- machine-readable benchmark run artifacts
- the default suite entry point for deterministic smoke validation

Stable benchmark families
-------------------------

- ``diffusion``: baseline adoption curves such as Bass, logistic, and Gompertz
- ``substitution``: replacement-share benchmarks such as Fisher-Pry and Norton-Bass
- ``competition``: multivariate competition benchmarks for stable product comparisons

Submodules
----------

.. toctree::
   :maxdepth: 4

   innovate.benchmarks.corpus
   innovate.benchmarks.model_cards
   innovate.benchmarks.runner

Module contents
---------------

.. automodule:: innovate.benchmarks
   :members:
   :show-inheritance:
   :undoc-members:
