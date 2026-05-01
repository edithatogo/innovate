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
- fast metadata checks for benchmark contribution review
- a canonical runner that emits comparable metrics, diagnostics, and
  uncertainty summaries
- JSON-friendly artifacts that can be saved and diffed in CI

Fast validation
---------------

Run the fast validation gate before adding or changing benchmark cases:

.. code-block:: python

    from innovate.benchmarks import (
        refresh_model_card_summaries,
        validate_benchmark_corpus,
    )

    report = validate_benchmark_corpus()
    report.assert_valid()

    summaries = refresh_model_card_summaries()
    print(summaries["bass"]["freshness"]["status"])

This gate checks required metadata, model-card freshness, and CI policy. It is
intended for normal pull request CI and does not execute timing benchmarks.

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

1. Run ``validate_benchmark_corpus`` after editing cases or model cards.
2. Use ``workflow_dispatch`` for opt-in timing runs.
3. Save the JSON artifact as a release or CI output.
4. Use the model cards to confirm assumptions, outputs, diagnostics, and
   limitations before interpreting the scores.
5. Keep documentation synchronized with code changes so the suite stays
   reproducible and auditable.

Promotion metadata
------------------

Backend and Rust-core promotion candidates must report reference backend timing
separately from accelerated results. XLA compilation cost and XLA
steady-state runtime should be recorded independently so first-call compilation
does not get confused with repeated execution. Cases that require expensive
accelerator timing should use ``workflow_dispatch`` or scheduled CI instead of
the fast default test path.
