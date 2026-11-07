# Historical policy wrapper review

Recovered PolicyHazardDiffusionModel and PolicyTimingInputs from 56d91dd7.
Native policy.rs delegates baseline fitting/prediction to Rust scalar models,
holds decay fixed during baseline fitting, and applies summed decaying event
level multipliers. Event boundaries are inclusive. Negative adjusted levels are
clipped and historical cumulative maxima are batch-local. Sample increments are
not time derivatives. Metadata explicitly excludes calibrated survival-hazard
and causal interpretations. Independent analytic events, fixed-decay fitting,
held-out state forecasts, summaries and invalid inputs are covered by four tests.

R singleton event vectors remain arrays; known node/covariate label arrays use
the same transport preservation while named input covariate maps retain their
separate handling. Installed Python Polars fitting and forecasts, R singleton
boundaries and Julia boundaries pass. Unknown payload fields and unsupported
stochastic/intervention/base-covariate paths fail explicitly.

Current scope supports scalar Bass/Logistic/Gompertz/Fisher-Pry baselines. Full
base composition/covariates, estimating effects/decay, stochastic uncertainty,
causal validation and richer policy scenarios remain required. Adjusted training
predictions can differ from observations because historical baseline fitting
ignores policy multipliers; diagnostics retain the baseline result and warn.

Validation: 71 Rust tests (67 core/C ABI coverage-run plus four Polars), 30
installed Python checks, 147 Julia assertions and R native conformance. Core/C
ABI line coverage: 81.99%. Clippy, lint/types, 163-page docs, package metadata and
hooks pass within local scope. Dependencies unchanged; previous audit evidence
and its unsuppressed bincode maintenance warning remain applicable. Clean-source,
platform/hosted packages and full scientific qualification remain open. All 16
models and 22 feature obligations stay required. No main promotion/publication
or native tenant operation occurred; full Conductor goal remains active.
