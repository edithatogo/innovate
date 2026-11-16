# Tutorial coverage audit and proposed curriculum

## Scope and baseline

This is the DOC-02/04/06/07 curriculum for the
[learning programme](learning-plan-20261116.md), inspected at
06b16671b654b4e51f158bf7d605c4286f156755 on 16 November 2026.
The following counts describe the maintained source inventory, not every page on
the public site or every notebook in the repository. Proposed lessons are pending.

Authoritative inspected records:

- [Content and executable-example contract](../../../docs/contracts/content-quality.json).
- [Astro configuration](../../../docs/astro-site/astro.config.mjs) and
  [content schema](../../../docs/astro-site/src/content.config.ts).
- [Native model discovery at the inspected revision](https://github.com/edithatogo/innovate/blob/06b16671b654b4e51f158bf7d605c4286f156755/rust/crates/innovate-core/src/discovery.json).
- [Generated polyglot API inventory](../../../docs/contracts/polyglot-api-documentation.json).

The content contract classifies **87 hand-authored pages**: 5 index, 61 historical,
13 reference and **8 tutorial** pages. Generated API pages have their own inventory.
The tutorial pages use **6 distinct example sources**; the contract registers
**10 Python examples** overall. A shared example used by two pages does not count
as two distinct workflows. All ten are configured for offline execution; none
registers an empirical public dataset.

| Current tutorial page | Registered example | Coverage implication |
| --- | --- | --- |
| `user-guide/getting-started.mdx` | `native_forecast.py` | First native Logistic result |
| `user-guide/forecasting.mdx` | `native_forecast.py` | Same source, additional forecast explanation |
| `tutorials/scenario-experiment-workflows.mdx` | `native_scenarios.py` | Scenario workflow |
| `tutorials/arrow-interchange.mdx` | `native_scenarios.py` | Same source, interchange explanation |
| `tutorials/norton-bass.mdx` | `native_norton_bass.py` | Generations/variant semantics |
| `tutorials/bayesian-fitter.mdx` | `native_bayesian.py` | Bounded native inference example |
| `tutorials/counterfactual-analysis.mdx` | `native_overlap.py` | Overlap/counterfactual teaching |
| `tutorials/data-ingestion-provenance.mdx` | `native_reingestion.py` | Re-ingestion/provenance contract |

The remaining `native_kairos.py`, `native_forcing.py`, `native_network.py` and
`native_batch.py` examples appear in the broad native reference, rather than in
focused standalone tutorials. This is useful material to promote and explain.

The current source has been substantially updated toward native execution, but
coverage is not yet adequate for a complete beginner-to-advanced empirical path.
In particular, a newcomer needs a continuous route from observed data through
fitting, comparison, chronological assessment, diagnostics and interpretation.
Advanced reference content and generated API documentation do not establish that
this learning route exists or that a reader can complete it.

### Legacy notebooks are a separate population

The three inspected root notebooks — `innovate_comprehensive_demo.ipynb`,
`innovate_comprehensive_demo_fixed.ipynb` and `innovate_examples.ipynb` — contain
legacy `innovate.diffuse` / `ScipyFitter` and pandas-era patterns. They do not count
as current native tutorial coverage. DOC-02 must inventory the remaining old
notebooks before deciding which scientific lessons deserve migration.

For each notebook record its imports, intended lesson, data origin, outputs,
current replacement and disposition: migrate, retain as historical, or retire
through a reviewed change. Inspect output material before republication. A format
conversion to marimo neither updates its API nor validates its scientific claims.

## Model-family coverage target

The native discovery record exposes these **sixteen exact keys**. The table is
a proposed teaching allocation, not a declaration that every operation has passed
scientific acceptance. Existing shared examples must be traced to their actual
model and variant before assigning coverage credit.

| Model key | Proposed lesson route | Required distinction or limitation |
| --- | --- | --- |
| `bass` | B04, I06, A02 | Aggregate curve versus finite-population Kairos engines; parameters depend on time and observation units |
| `fisher_pry` | B04 | Substitution shares, endpoint handling and assumptions about the denominator |
| `gompertz` | B04 | Asymmetric curve, parameterization and supported historical/ODE variants |
| `logistic` | B01–B04, I08 | Existing forecast foundation; add observed-data fitting and uncertainty interpretation |
| `norton_bass` | I02 | `legacy_cannibalization` versus `successive_generations`, launch times and rate/level quantities |
| `lotka_volterra` | I03 | Competition equations, initial states, parameter domains and observable outputs |
| `multi_product` | I03 | Coupled products, output ordering and conserved/nonconserved quantities |
| `complementary_goods` | I03 | Complementarity mechanism and interpretation of interaction parameters |
| `lock_in` | A03 | Path dependence, starting conditions and finite-window identifiability |
| `network_diffusion` | A01 | Deterministic spillover family; distinguish it from Bass with `kairos_network` |
| `policy_hazard` | I06, A05 | Declared policy-level multiplier semantics; distinguish it from Bass forcing and causal estimation |
| `latent_process` | I05, A03 | Current residual-path semantics; avoid implying a general latent-state posterior |
| `regime_switching` | I06, A03 | `historical_index` versus `fixed_time`; current piecewise curves versus an inferred hidden Markov process |
| `hierarchical` | I04 | Group point curves and `historical_additive`/`centered` variants; distinguish Bayesian partial pooling |
| `mixture` | I05 | `gaussian_residual` versus `historical_adoption`, component ambiguity and prediction meaning |
| `composite` | A04 | Coupled growth ODE, `fixed`/`historical_batch`, exposed model controls and documented solver behaviour/limits |

All sixteen records currently carry experimental stability. A discovery flag is
not a substitute for the owner-qualified operation/variant contract. In particular,
family-level `supports_rates: false` for Bass, Gompertz and Logistic needs
reconciliation with quantity-specific native paths before generating a public
capability table. Record that ambiguity; do not silently choose one interpretation.

## Curriculum

IDs below are content work items within the existing documentation track, not new
Conductor tracks or promises of one notebook per row. Reuse canonical examples
and thin notebook adapters. Dataset IDs resolve to the
[admission shortlist](public-datasets-20261116.md); none is admitted by this plan.
Controlled fixtures remain essential for independent expected results.

### Beginner: complete one defensible analysis

| ID | Lesson and learning outcome | Prerequisite | Data and executable scope |
| --- | --- | --- | --- |
| B01 | Install, discover capabilities and obtain an interpreted first result | Basic language use | Existing Logistic fixture; equivalent Python, Julia and R routes; N01 optional marimo interface |
| B02 | Prepare observations: dates, units, fractions, missingness, duplicates and stock/flow distinctions | B01 | Controlled malformed cases and D02/D03 excerpts; Polars preparation through supported inputs |
| B03 | Fit the first curve, inspect diagnostics and evaluate a chronological holdout | B02 | D01; native fit/predict/summarize/diagnose, baseline comparison and honest poor-fit outcome; N02 after DOC-03 |
| B04 | Choose among Bass, Logistic, Gompertz and Fisher–Pry for the same question | B03 | D01 plus independent controlled curves; common target, units, split and scoring rules |
| B05 | Save and reproduce an analysis with data and result provenance | B03 | D01 result bundle, source/native identity, Arrow IPC where supported, checksums and replay |

The beginner route must show the complete small input and expected output, define
every parameter used, and explain what a successful fit does and does not show.
Temporal assessment must prevent future information from entering preprocessing,
model choice or tuning. Assess held-out error and a simple reference forecast;
training fit alone is not predictive evidence. Respect percentage/fraction and
time-origin conventions and explain out-of-range or nonmonotone observations.

Choose calendar aggregation by observation quantity: sum compatible interval
flows, but define an end-of-period or average measure for stocks and capacity.
Missing observations must remain distinguishable from observed zeros. The native
aggregation helper's summation/zero-fill behaviour does not establish an appropriate
observation model or prove that no activity or stock existed in an absent period.

### Intermediate: choose the right observation and model structure

| ID | Lesson and learning outcome | Prerequisite | Data and executable scope |
| --- | --- | --- | --- |
| I01 | Analyse several related observed series without confusing stocks and flows | B04–B05 | D03 fuel-type stocks and first registrations; per-series fitting/comparison before structural substitution claims |
| I02 | Model successive generations and interpret cannibalization and launch times | B04 | Existing Norton–Bass fixture; compare both variants and appropriate rate/level outputs |
| I03 | Specify competition and complementarity | I02 | Independent controlled cases for Lotka–Volterra, multi-product and complementary goods; limits on empirical identification |
| I04 | Compare grouped trajectories and explain what sharing means | B04–B05 | D01 countries; current hierarchical variants and independent fits; no implied partial pooling |
| I05 | Interpret mixtures and latent contributions | B04 | Controlled separated/ambiguous components; current mixture and latent-process semantics |
| I06 | Work with calendar time, covariates, forcing, parameter events and regimes | B04 | D02 plus controlled events; separate `policy_hazard`, Bass forcing and regime APIs; scenario assumptions explicit |
| I07 | Execute many reproducible fits and retain per-group failures | B05, I04 | D01 country batches and existing batch fixture; bounded workers, deterministic ordering and partial failures |
| I08 | Quantify uncertainty and check calibration | B03 | Existing analytic Bayesian fixture plus noisy controlled data; supported posterior/interval methods and owner-qualified NUTS or bootstrap routes |
| I09 | Summarize adoption timing without conflating weights and elapsed time | B02–B03 | Controlled irregular-time observations; supported category/summary operations and their integration conventions |

I08 must distinguish parameter uncertainty, predictive uncertainty, observation
noise and scenario variation. A convergence statistic alone does not establish
calibration or fit. A bootstrap or NUTS lesson is published only for the actual
accepted native path; the tutorial must not implement a missing inference engine.

### Advanced: mechanisms, difficult inference and portable workflows

| ID | Lesson and learning outcome | Prerequisite | Data and executable scope |
| --- | --- | --- | --- |
| A01 | Use measured contact structure and explain network semantics | I03, B05 | D04 topology with synthetic adoption; deterministic `network_diffusion` versus Bass/Kairos network execution |
| A02 | Design controlled simulation/scenario experiments | I06 | Existing Kairos/forcing/network fixtures; engines, seeds, clocks, resources, repeated runs and scenario comparison |
| A03 | Investigate lock-in, latent trajectories and regime ambiguity | I03, I05–I06 | Controlled weak/strong-information cases; initial states, histories and alternative explanations |
| A04 | Configure composite ODEs and diagnose solver limits | I03 | Independent coupled-system fixtures; exposed model controls, documented solver behaviour/fixed limits, rates and explicit failure handling |
| A05 | Distinguish intervention scenarios from supported counterfactual estimates | I06, I08 | Existing overlap example; estimand, overlap, assumptions and unsupported cases; optional D05 only after separate admission |
| A06 | Diagnose difficult fits and assess local identifiability | I08, A04 as relevant | Controlled weak-information data; starts, multistart/local-rank diagnostics, bounds and failed fits through supported APIs |
| A07 | Migrate a legacy workflow and replay results across bindings | B05 and relevant model lesson | Current Python/Julia/R bundles, IPC/JSON semantics and explicit non-equivalences |

An empirical example need not be invented for every structural family. If the
available dataset cannot identify a mechanism, teach the model with a controlled
case and explain the evidence that an empirical application would need.

## Operation, language and scientific coverage

DOC-02 must enumerate public operations from the maintained native/polyglot
contracts, including supporting discovery, preparation, fitting, prediction,
simulation, summaries, diagnosis, inference, scenarios, batch and interchange
surfaces. For each record attach model/variant/quantity, input assumptions, current
reference, example/tutorial, binding support and qualification evidence.

Use a model × variant × operation × quantity matrix, with dispositions such as
executed, pending qualification, unsupported with demonstrated rejection, or
historical. Count applicable combinations explicitly. Do not manufacture every
possible Cartesian combination or infer support from a family-wide boolean.

The learner-facing model card should state: question addressed, equation and
parameterization, domains and units, observation assumptions, required inputs,
variants/options/defaults, returned quantities, diagnostics, failure cases,
primary scientific references and current executable routes. Each equation and
scientific claim receives a scoped owner review.

Python, Julia and R each need a complete installation/first-result path and
examples for the operations actually exposed by that binding. Tabs represent
equivalent tasks only after per-language execution. Marimo provides a Python
interface; it does not by itself extend notebook support to Julia or R.

## Definition of adequate and comprehensive coverage

The target is maximal useful coverage of the supported library surface:

1. A complete B01–B05 route produces an interpreted, reproducible empirical result.
2. All sixteen discovered family keys have a current model card and an appropriate
   teaching route; newly discovered keys fail the unclassified-coverage check.
3. Every exposed supported operation has a checked API/example route and a
   meaningful explanation. Unsupported and unqualified cases remain visible.
4. Every advanced row has prerequisites, expected results, diagnostic interpretation
   and limitations; at least one relevant invalid-input or failure case is taught.
5. At least three admitted empirical sources cover scalar, calendar and multiseries
   analysis. D04 additionally teaches observed network inputs with synthetic outcomes.
6. The three maintained languages have qualified starting routes and accurate
   operation coverage, rather than untested presentation parity.
7. Human readers complete representative beginner and advanced tasks; specialist
   review assesses scientific interpretation separately from code execution.

Report these denominators separately. Page count, notebook count, a green build,
or synthetic recovery alone is not an adequate curriculum score.

## Keeping the lessons current

- Generate capability and related-content views from existing authoritative
  contracts; validate tutorial metadata and source bindings in the current build.
- Associate outputs with exact source, native artifact, environment and dataset
  identity. Re-run affected examples when those inputs change.
- Run normal examples offline on frozen fixtures. Dataset updates and dependency
  changes are separate reviewable increments with a recorded semantic comparison.
- Preserve current/historical/version labels in navigation and Pagefind. Legacy
  content never contributes to current coverage solely because its URL exists.
- Record the last executed revision and review state per lesson; avoid manually
  maintained freshness badges that can drift from the actual evidence.
