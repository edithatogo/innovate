# Historical Norton Bass fixed events review

Historical Norton Bass now accepts the same fixed parameter event contract as
the four ecosystem ODE families. Validation, persistent indexed overrides and
continuous-state segmentation share one Rust module. Name lookup is compiled
once. Unforced parameter vectors remain borrowed, and overrides are copied only
for active regimes. The module retains finite/nonnegative time, strict ordering,
parameter-domain and event-count checks.

Ecosystem integration delegates to the shared module. Lock-in retains capacity
checks for both sides of each used boundary. Historical Norton Bass composes
events with named generation controls and disables its single-generation
closed-form shortcut when events are present. Rates use the right-hand regime.
Saved state retains the complete history, initial conditions and event list.

Analytic two-generation and scalar zero-innovation cases verify continuity,
boundary rates and forecast behavior. An independent interacting DOP853
reference splits at off-grid events and control knots. A perturbed-start
conditional fit and held-out forecast pass. Python exercises Polars fitting
and the normal fitted forecast method. Julia and R fit the same reference.
Discovery advertises fixed events for the historical Norton Bass variant.

The successive-generation closed-form path retains its existing tests and
rejects nonempty parameter events explicitly. Its forcing/event extension,
event-time/override inference, original threshold helpers, uncertainty,
branch/mutation and scientific/platform release gates remain open.

The full core/C ABI and Polars tests pass. Current line coverage is 99.24 percent
for shared events, 90.50 percent for substitution and 93.93 percent for
competition. All 38 release obligations remain required.
