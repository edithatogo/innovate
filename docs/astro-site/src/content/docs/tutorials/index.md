---
title: Tutorials and learning routes
description: Choose an executable native tutorial by the scientific task and evidence you need.
---

These tutorials use the current Rust and Polars development API. Install the
[locked development environment](/innovate/user-guide/installation/) first.
Each current tutorial displays a checked example file, its expected output,
interpretation and limits. Run the unchanged file successfully before adapting
it to a new analysis.

## Choose a starting point

| Reader task | Tutorial | What its assertions establish |
| --- | --- | --- |
| Make a first native call | [Quickstart](/innovate/user-guide/getting-started/) | Logistic values, retained identifiers and rejection of duplicate times |
| Define prediction units and quantities | [Prediction and forecasting](/innovate/user-guide/forecasting/) | Agreement with a closed-form curve and the table boundary |
| Recover a known multi-generation model | [Norton–Bass fitting](/innovate/tutorials/norton-bass/) | Six-parameter recovery and held-out levels and signed derivatives |
| Inspect a sampler using a known posterior | [Bayesian fitting](/innovate/tutorials/bayesian-fitter/) | Support, declared posterior tolerances and chain diagnostic bounds |
| Compare specified policy and product scenarios | [Scenario experiments](/innovate/tutorials/scenario-experiment-workflows/) | Scenario invariants, saved artifacts and deterministic replay |
| Understand a cross-fitted contrast | [Counterfactual and causal analysis](/innovate/tutorials/counterfactual-analysis/) | Hand-calculated scores and rejection of invalid overlap/fold designs |
| Save and recover execution data | [Arrow round trips](/innovate/tutorials/arrow-interchange/) | JSON and Arrow recovery, row identity and supported replay |
| Detect altered dataset content | [Data ingestion and provenance](/innovate/tutorials/data-ingestion-provenance/) | Both declared checksum encodings and rejection of modified data |

## Work through a complete analysis

For a first modeling workflow, follow **quickstart → fitting → forecasting →
scenario experiments**. For a reproducible data workflow, follow **ingestion →
Arrow round trips → scenario replay**. For inference, read the
[fitting guide](/innovate/user-guide/fitting/) before choosing a Bayesian or
causal tutorial: they address different estimands and assumptions.

Use each tutorial's reader task to explain the result in your own words. A useful
handoff includes the quantity, units, inputs, model assumptions, relevant
diagnostics, failure behavior and the inference the example does not support.
Changing data or assumptions creates a new analysis requiring its own checks.

## Qualification and earlier material

Passing a synthetic example establishes its stated software checks. It does not
establish empirical accuracy, causal identification, interval coverage or
suitability for a particular population. The
[documentation quality guide](/innovate/maintainers/documentation-quality/)
separates structure, execution and broader scientific and reader-task review.

Earlier tutorials remain in the [versioned archive](/innovate/latest/tutorials/).
Some current-path specialist pages also retain an explicit pre-0.7 notice while
their replacements are being qualified. Use those notices to recognize earlier
interfaces; the [native capability guide](/innovate/native-migration/) determines
which capabilities are currently implemented and which gates remain open.
