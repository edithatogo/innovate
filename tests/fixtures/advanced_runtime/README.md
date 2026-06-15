# Advanced Runtime Fixtures

These fixtures use seed 20260614 and deterministic cumulative adoption values.
They are small enough for unit and CI smoke tests while preserving the workflow
shape needed by the advanced modeling runtime track.

## Regime ensemble

The regime ensemble case contains a visible pre/post period 4 adoption shift.
It is intended for ensemble composition and scoring tests without requiring an
optional changepoint dependency.

## Policy scenario

The policy scenario case turns on a rebate intervention at period 4. The
machine fixture keeps the policy indicator and media spend as auditable
covariates so scenario summaries can report their assumptions.

## Streaming update

The streaming update case separates an initial four-period fit window from two
later periods. This supports incremental-update tests with deterministic batches.

## Uncertainty calibration

The uncertainty calibration case splits calibration periods from holdout
periods. It is designed for interval coverage and residual diagnostic tests.
