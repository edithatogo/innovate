.. _benchmark_workflows_tutorial:

Benchmark Workflows and Model Cards
===================================

The benchmark suite provides a stable, machine-readable way to compare the
canonical model families shipped with ``innovate``. It keeps the benchmark
corpus, model cards, and evaluation outputs synchronized so the library can be
used for scientific comparison and release validation.

What the suite includes
-----------------------

- a reproducible benchmark corpus with stable case identifiers
- synchronized model cards for stable model families
- a canonical runner that emits comparable metrics, diagnostics, and
  uncertainty summaries
- JSON-friendly artifacts that can be saved and diffed in CI

Running the stable suite
------------------------

.. code-block:: python

    from innovate.benchmarks import run_stable_benchmark_suite

    suite = run_stable_benchmark_suite()
    print(suite.to_dict()["run_count"])

    for run in suite.runs:
        print(run.case_id, run.model_key, run.metrics["RMSE"])

Saving benchmark artifacts
--------------------------

.. code-block:: python

    from pathlib import Path
    from innovate.benchmarks import run_stable_benchmark_suite

    output_dir = Path("benchmark-results")
    output_dir.mkdir(exist_ok=True)

    suite = run_stable_benchmark_suite(model_keys=("bass", "fisher_pry"))
    suite.write_json(output_dir / "stable-suite.json")

Interpreting outputs
--------------------

- ``metrics`` contains the comparable fit measures for each benchmark run.
- ``diagnostics`` records the standardized diagnostics contract, including
  support level, warnings, and residual analysis.
- ``uncertainty`` describes whether the result is deterministic, bootstrap, or
  Bayesian and includes the provenance required to compare runs safely.
- ``metadata`` captures the stable model card and benchmark case identity so
  artifacts can be traced back to the corpus version used for the run.

Model-card synchronization
--------------------------

The model-card registry is generated from the stable capability registry, so
each stable family has a consistent machine-readable description.

.. code-block:: python

    from innovate.benchmarks import get_model_card, list_model_cards

    cards = list_model_cards()
    bass = get_model_card("bass")

    print(sorted(cards))
    print(bass.summary)
    print(bass.benchmark_case_ids)

Recommended workflow
--------------------

1. Run the stable benchmark suite for the model families you want to compare.
2. Save the JSON artifact as a release or CI output.
3. Use the model cards to confirm assumptions, outputs, diagnostics, and
   limitations before interpreting the scores.
4. Keep documentation synchronized with code changes so the suite stays
   reproducible and auditable.
