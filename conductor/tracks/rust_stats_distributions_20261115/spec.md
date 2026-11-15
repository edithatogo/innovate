# Specification: Rust Statistical Distributions and Quantiles Modernization

## Objective
Replace hand-rolled statistical distribution and quantile helpers (such as `bayesian::normal_quantile` and log-PDF/CDF/normal/Student-t helpers in `bayesian.rs` and `causal.rs`) with `statrs` from `crates.io`.

## Scope
- Replace `normal_quantile` and custom distribution density/CDF calculations with `statrs::distribution` (`Normal`, `StudentsT`, `ChiSquared`, `Beta`, `Gamma`, `Poisson`).
- Preserve existing Vehtari split-R-hat (`ranked_rhat`, `split_rhat_moments`) and bulk-ESS (`paired_ess`) conventions and exact test tolerances.
- Optionally evaluate `nuts-rs` only as a separate follow-on if HMC/NUTS sampling replaces custom MCMC proposals.

## Audited implementation scope — 15 November 2026

Base: `42189a223fd44c823ca11aec689ea67b11c25790`. The approved bounded
assignment and [ownership record](ownership.md) constrain this lane; broader
track objectives remain subject to their owners and acceptance prerequisites.

| Location | Actual calculation | Decision |
| --- | --- | --- |
| `bayesian::normal_quantile` | 48-step bisection of `0.5*libm::erfc(-x/sqrt(2))` over `[-8,8]` | Replace with the exact pinned `statrs::distribution::Normal` inverse CDF. |
| `GaussianPriors::log_density` | Sum of unnormalised `-0.5*((value-mean)/sd)^2` scores | Preserve arithmetic; independent near-centre and extreme-scale fixtures prohibit normalizer-subtraction cancellation. |
| `bayesian::log_likelihood` | Same unnormalised score for actual native predictions and fixed known noise | Preserve arithmetic, validation, aligned observations and nonfinite score behavior. |
| `bayesian::grid` | Cumulative discrete posterior mass quantiles | Retain discrete equal-tail convention; not a normal-distribution quantile. |
| `bootstrap::fit` | Empirical parameter/forecast type-7 percentiles through `numerics::linear_quantile`, scaled sample standard errors | Preserve methods, RNG, failure counts and null intervals; no analytic PDF/CDF helper exists here to migrate. |
| `causal.rs` (read only) | Fixed `1.96` interval multiplier, sensitivity formulas, logit/log-likelihood expressions, empirical bootstrap | No standalone normal/Student-t CDF/PDF/quantile helper was found. The fixed multiplier and all causal changes remain integrator-owned. |

The audit searched the plan's named symbols and inspected the actual callers,
including exponential/logarithmic expressions. `gamma` sensitivity parameters
and `counterfactual` substring matches are not gamma-distribution or error-function
implementations. No Beta, Gamma, ChiSquared, Poisson or Student-t implementation
is invented solely to use a dependency named in the original scope.

### Private quantile domain and preservation

The only caller is `rank_scores`. Public diagnostics permit 2–16 finite chains
with 4–10,000 equally sized draws. Split chains pool at most 160,000 values.
Average tied ranks and Blom's formula bound every attainable probability by
`0.625/160000.25` and `1-0.625/160000.25` (approximately `3.90624e-6` and
`0.999996093756`). These are inside the old search interval. Preserve average
tie ranks, signed-zero equivalence, odd-middle exclusion, folded medians,
`ranked_rhat`, `split_rhat_moments` and `paired_ess` unchanged.

The old saturation for impossible private probabilities is recorded separately
in the oracle fixture, not promoted to a public API. Direct dependency tests
cover mathematical endpoints, nonfinite arguments and checked inverse-CDF
errors without claiming those operations are exposed by Innovate.

### Baseline and tolerances

The independent mpmath fixture uses 400 decimal digits and exact binary64
input values, with generator/input hashes and predeclared tolerances. Ordinary
and tail quantiles use absolute `2e-10`; CDF/survival tails use zero absolute
and `2e-13` relative tolerance. Relative Gaussian scores retain nonzero tiny
values using zero absolute and `5e-15` relative tolerance. The original R-hat
`1e-12`, posterior mean `1e-10` and bootstrap tests remain unchanged.

Separate main-produced behavioral receipts pin seeded sample draws, accepted
counts and posterior masses; they are explicitly not independent mathematical
oracles. Bootstrap interval tests use independent fixed type-7 coefficients
on returned estimates, so this lane does not freeze the separately owned fit
solver's internal numerical outputs.

Full shared dependency, causal, workspace, installed-consumer, security, coverage
and scientific-release acceptance remain open. NUTS and planning-only downstream
tracks are not activated by these fixtures or a source migration.
