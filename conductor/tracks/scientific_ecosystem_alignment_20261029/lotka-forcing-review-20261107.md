# Historical Lotka Volterra forcing review

The native path preserves the four historical named additive coefficients for
each declared control. Base rates remain nonnegative and forcing coefficients
are signed. Saved history uses the shared native interpolation, endpoint hold
and revision checks already qualified for Bass, Gompertz and Norton Bass.
Integration includes every control knot, and the returned state retains the
origin, initial values and complete control history.

The analytic case tests levels and instantaneous rates for uncoupled growth.
The independent interacting reference splits DOP853 at each control knot and
checks both levels and rates. A perturbed conditional least-squares start
converges against that reference and forecasts held-out times. The installed
Python consumer exercises Polars fitting. Julia and R fit the same independent
reference and forecast using saved history.

Review checked signed beta transforms, complete parameter names, forecast
origin and revised-history rejection. All other competitive families retain
their existing parameter restrictions. The full core/C ABI and Polars suites
pass. Overall line coverage is 85.37 percent and competition line coverage is
92.25 percent. Branch/mutation and wider scientific qualification remain open.

This checkpoint conditions on forcing and initial state. Automatic forcing-fit
starts, event splits, weighted fits and calibrated uncertainty remain required.
No release obligation is closed by this bounded checkpoint.
