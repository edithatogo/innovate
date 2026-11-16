---
title: Core concepts
description: Choose a native operation, define the quantity and units, and understand what its result establishes.
---

Innovate's 0.7 development API executes scientific operations in Rust and uses
Polars and Arrow for tabular input and output. The language interface is a
transport and usability layer. Begin with the [installation instructions](/innovate/user-guide/installation/)
and [checked quickstart](/innovate/user-guide/getting-started/).

## Choose the scientific task

| Task | Current entry point | What you supply | What you must still establish |
| --- | --- | --- | --- |
| Evaluate a specified curve | `predict` | Model, parameters and an explicit time table | Whether those parameters and that model describe the population |
| Estimate parameters | `fit` | Ordered observations, model options and fitting controls | Identifiability, fit quality and sensitivity to assumptions |
| Generate a stochastic trajectory | A supported simulation operation through `execute` | Engine, initial state, time grid and seed | Whether the stochastic mechanism and calibration are suitable |
| Compare specified scenarios | `execute_scenario_grid` | Complete baseline and alternative specifications | Whether the scenario change represents a credible intervention |
| Summarize a causal contrast | A supported causal operation through `execute` | Outcomes, treatment, design information and required nuisance estimates | The identification assumptions behind the estimand |

The [current API reference](/innovate/api/python/) records exact signatures.
The [kernel guide](/innovate/core/kernel/) explains requests, responses and errors.
A scenario difference and an identified causal effect answer different
questions; the [counterfactual tutorial](/innovate/tutorials/counterfactual-analysis/)
shows where that distinction matters.

## Define the quantity before selecting a model

A **level** is the model's state at a time. A **rate** is its native derivative
where supported. A count of events during an interval requires an interval
definition and an appropriate event or flow model. These quantities are not
interchangeable.

For successive product generations, a generation's level can decline as users
transfer to another generation. Its signed derivative is a net change in that
level. Calling every level “cumulative adoptions” would incorrectly imply that
each must increase forever. Work through the [Norton–Bass example](/innovate/tutorials/norton-bass/)
before interpreting a multi-generation output.

## Keep units and identities explicit

Time is a finite numeric coordinate in units selected by the caller. A parameter
measured per year must be rescaled when a time grid is expressed in months.
The table boundary does not infer a unit conversion from a column name.
Record the population, time origin, time unit and meaning of the observed value
alongside the analysis.

The table prediction interface requires a nonempty, strictly increasing `time`
column without nulls. It preserves input columns and row order and appends
predictions. Pass an identifier column when you need to reconcile predictions
with other data. A `LazyFrame` is collected at execution, and the current IPC
boundary copies buffers. See the [interchange contract](/innovate/core/arrow-interchange/).

## Use evidence at the right level

| Observation | Supported conclusion | Further evidence needed |
| --- | --- | --- |
| A request passes validation | It satisfies the implemented input rules | Measurement validity and scientific suitability |
| An optimizer reports convergence | Its stopping criterion was met | Identifiability, alternative starts and model adequacy |
| A seeded run repeats | That supported configuration is replayable in the checked environment | Compatibility across versions and empirical calibration |
| An example matches a closed-form answer | The checked case agrees with an independent calculation | Coverage of other parameter regions and real datasets |
| An interval is returned | The selected method produced an interval | Calibration, assumptions and coverage for the intended setting |

The [native capability guide](/innovate/native-migration/) records implemented
operations and open scientific gates. Documentation execution and scientific
acceptance are reported separately in the [quality contract](/innovate/maintainers/documentation-quality/).

## A practical learning route

First verify the [prediction contract](/innovate/user-guide/forecasting/). Then
review the [fitting guide](/innovate/user-guide/fitting/) and recover a known
synthetic model. Next choose between [scenario experiments](/innovate/tutorials/scenario-experiment-workflows/),
[Bayesian diagnostics](/innovate/tutorials/bayesian-fitter/) and
[dataset provenance](/innovate/tutorials/data-ingestion-provenance/) according to
your task. Keep the expected result and failure case visible when adapting an
example to your own analysis.

The [earlier core page](https://github.com/edithatogo/innovate/blob/6c226f92f14f01e771b1490c7fb3a7c77f4bc74a/docs/astro-site/src/content/docs/latest/core/index.md) remains in the source history.
