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
- fast metadata validation with ``validate_benchmark_corpus``
- model-card freshness summaries with ``refresh_model_card_summaries``

Stable benchmark families
-------------------------

- ``diffusion``: baseline adoption curves such as Bass, logistic, and Gompertz
- ``substitution``: replacement-share benchmarks such as Fisher-Pry and Norton-Bass
- ``competition``: multivariate competition benchmarks for stable product comparisons

Fast and opt-in automation
--------------------------

Fast CI should validate benchmark metadata and model-card freshness without
running expensive timing suites:

.. code-block:: bash

   uv run python -m pytest tests/unit/test_benchmark_automation.py

The expensive benchmark suite remains opt-in through ``workflow_dispatch`` and
uses:

.. code-block:: bash

   uv run pytest --benchmark-only --benchmark-json=benchmark.json

Benchmark metadata records the CI policy, runtime tier, reference backend, XLA
compilation cost requirement, XLA steady-state runtime requirement, accelerator
target, and baseline model key. These fields keep fast checks bounded while
preserving the evidence needed for Rust-core and optional-backend promotion
decisions.

MARS surrogate benchmark gate
-----------------------------

The MARS surrogate benchmark gate keeps ``mars`` optional and unpromoted until
metadata and opt-in benchmark evidence justify an adapter. Fast CI validates
only the gate metadata. Opt-in artifacts compare the NumPy/SciPy reference path,
the MARS surrogate candidate, and the eligible JAX/XLA alternative while
recording whether any gain came from the surrogate, XLA, their interaction, or
neither.

Submodules
----------

.. toctree::
   :maxdepth: 4

   innovate.benchmarks.corpus
   innovate.benchmarks.automation
   innovate.benchmarks.model_cards
   innovate.benchmarks.mars_surrogate
   innovate.benchmarks.runner

Module contents
---------------

.. automodule:: innovate.benchmarks
   :members:
   :show-inheritance:
   :undoc-members:
