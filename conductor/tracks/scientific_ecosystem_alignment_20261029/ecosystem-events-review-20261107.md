# Fixed ecosystem parameter events review

Lotka Volterra, multi-product, complementary goods and lock-in now support
persistent fixed named overrides. The list is validated for parameter names,
base/beta domains, finite increasing nonnegative times and at most 32 entries.
Events at or before the origin select the initial active regime. The supplied
or conditioned initial state remains the trajectory's starting state.

Integration inserts each used event boundary and freezes its parameter regime
through that interval, including the left integration endpoint at the event.
The resulting state carries continuously into the next interval. Reported
rates use the right-hand active parameters. Linear control history composes
with overrides of both base and beta parameters. Parameter vectors remain
borrowed when there are no active events.

Lock-in validates capacity on both sides of each used boundary, retains its
instantaneous reporting projection and preserves the integrated trajectory.
Fixed overrides can alter structural identifiability. Eventful lock-in therefore
reports that structural analysis is required instead of reusing the unforced
rank bound. Conditional fits keep event override values fixed, and diagnostics
record that they are not estimated.

Analytic cases check boundary continuity, right-hand rates, events before the
origin, invalid events and capacity decline/recovery. Independent interacting
DOP853 references split at two off-grid changes and every control knot for all
four models. Conditional fits, saved-state forecasts and normal Polars fitting
pass against those references. Discovery advertises the exact fixed-event
contract. R's scalar event times are preserved separately from observation
time arrays, with the native failure and corrected singleton case retained.

The full core/C ABI and Polars suites and Python/Julia/R conformance gates pass.
Event-time/override inference, other model families, uncertainty, automatic
initialization, branch/mutation and scientific/platform release qualification
remain open. All 38 release obligations remain required.
