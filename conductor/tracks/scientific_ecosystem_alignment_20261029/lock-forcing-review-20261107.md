# Lock in forcing review

The planned covariate extension uses named additive effects on all seven base
parameters and the existing shared history contract. Effective capacity is
validated as finite and positive at the origin and relevant interpolation
knots/sample times. Linearity between knots and endpoint holds preserve this
domain over the integration interval. Invalid capacity fails before clipping.

The historical RHS clips its inputs. Reported output now follows instantaneous
capacity while preserving the integrated trajectory. The decline/recovery
regression verifies that output clipping does not reset state. Returned rates
remain the historical RHS at clipped levels. They exclude the derivative of the
reporting projection, and that convention is explicit in metadata and docs.

Forced fits allow a base m below observations because controls can add to
capacity. Unforced fits preserve their previous market lower bound. The new
regression covers this distinction. Structural diagnostics now count all named
parameters and retain a confounding rank upper bound. For interior trajectories,
scaling capacity and feedback combinations together preserves the RHS even
with additive controls. Clipping can provide additional boundary information.

The independent DOP853 fixture checks interacting predictions and RHS rates.
A perturbed-start fit and held-out forecast pass. Python uses the Polars fit
path, and Python/Julia/R verify the capacity projection recovery. All affected
competitive, complementary and lock-in tests pass, as do the C ABI/Polars tests.
Competition line coverage is 93.31 percent from the current focused run.
Unchanged unrelated modules retain their earlier full-suite evidence.

Event splits, automatic starts, forcing and initial-state uncertainty,
calibrated parameter inference, branch/mutation and scientific/platform release
qualification remain open. All 38 release obligations remain required.
