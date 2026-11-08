# Historical Logistic and initial ABM review

Same isolated lane, based on 84eca121. Historical source was read directly from
56d91dd7: Logistic modifies L/k/x0 pointwise with additive covariates, while its
independent post-event curve applies at t >= t_event and ignores covariates.
The native port preserves this with explicit historical_ignored metadata and
fit warnings; an additive mode applies common coefficients to both segments.
Independent closed-form predictions and parameter recovery verify both branches,
the boundary, future forecasts and local rank. Advanced fits require explicit
initial parameters. No continuity, rate-helper or all-model parity is claimed.

Polars forwards declared covariate columns into Rust and retains row identity.
Review found a single-element R covariate vector being unboxed to a scalar;
named input covariate arrays are now preserved. The failing R receipt and the
installed-package rerun are retained. Unknown options, malformed covariate
shapes/names, invalid effective domains and inconsistent states fail explicitly.

Kairos simulation now accepts initial_adopters in 0..=population. Its identity
stream selects initial ABM adopters without perturbing the timing stream; counts
include initial agents at time zero. Tests cover partial and complete adoption,
identity/state conservation and previous simulation/forcing conformance.
Heterogeneous/network/resource ABM and resumable state remain required work.

Validation: 51 Rust tests (47 core/C ABI in coverage plus 4 Polars), 25 installed
Python checks, 137 Julia assertions and R native conformance. Clippy, lint/types,
163-page docs and package metadata pass. Core/C ABI line coverage is 80.75%.
Dependencies were unchanged; the preceding audit receipt remains applicable to
that graph, including its unsuppressed bincode maintenance warning. Clean-source
package, cross-platform and full scientific acceptance remain open. All 38 model
and feature obligations remain required; no track archival, main promotion,
external publication or native tenant operation occurred.
