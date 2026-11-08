# Successive generation forcing review

The forced successive-generation variant integrates each Bass fraction from
its own fixed launch in log-survival space. The old unforced closed forms
remain available. Effective innovation/imitation are finite nonnegative and
market capacity is finite positive after launch. Shared history begins at or
before the first launch. Events and control knots partition the integration,
and parameter overrides carry the fraction state continuously.

Successive inheritance maps fractions into levels. Capacity can vary between
control knots and jump at a fixed override. Reported rates include capacity
slopes and inheritance terms, use the active right-hand regime, and exclude
jump impulses. Capacity derivatives are computed only for reports. This
prevents a slope from the next interval from invalidating the previous
fraction integration. Scaling avoids intermediate overflow or underflow where
the capacity rate remains representable, including zero coefficients.

Analytic checks cover launch rates, capacity slopes, endpoint holds and a
capacity jump. Targeted regressions retain the finite-scale and first-launch
history failures and the mixed-boundary slope failure. The independent
DOP853 fixture integrates fractions directly through an off-grid launch and
two events. Predictions, rates, perturbed fitting and a held-out forecast
pass. Python exercises Polars fitting and saved forecasts for both Norton
variants. Julia and R fit and forecast the same fixtures.

The full core/C ABI and Polars tests pass. Substitution and mechanistic
line coverage are 91.33 and 90.68 percent respectively. Package checks are
local and confirm metadata, contents and exact scientific source inclusion.

Launch/event times and override values are fixed inputs. Their inference,
automatic starts, original threshold helpers, calibrated uncertainty,
branch/mutation and wider scientific/platform qualification remain required.
All 38 release obligations remain open.
