# Scientific integrity specification

## Problem

At baseline `b9998ad`, `src/innovate/fitters/bayesian_fitter.py` catches sampling errors, draws values around initial estimates and exposes them as posterior samples. `src/innovate/causal/policy.py` advertises estimator choices while inspected ATE/ATT paths use unadjusted differences; the ATT confounder argument is unused. These are scientific correctness risks, not missing marketing features.

## Contract

A failed sampler returns an explicit failed-inference state and diagnostics; posterior and credible-interval accessors reject that state. A separately requested deterministic estimate is labelled as such and never encoded as posterior draws. Distinguish prior, posterior, bootstrap, predictive and point-estimate outputs.

An estimator must implement its selected method and declared adjustment, or raise an unsupported-method error. Keep descriptive contrasts available under descriptive names. Require an estimand, target population, identification assumptions, adjustment variables, overlap checks and uncertainty method for causal-labelled outputs. A configured causal contract alone is not causal identification.

## Acceptance

Test failure propagation, stale results after a second failed fit, impossible/empty data, non-finite input, absent treatment groups, confounder use and unsupported choices. Validate likelihoods and gradients against independent references. Use simulation-based calibration, repeated-simulation interval coverage and deliberately misspecified examples for supported inference paths. Reconcile both Bayesian fitter implementations, kernel results and reports so no downstream consumer reintroduces the invalid label.

The release evidence must distinguish a verified descriptive estimator, an assumption-dependent causal estimate, and an invalid fit. No new causal method is required merely to repair misleading dispatch: explicit rejection is acceptable.
