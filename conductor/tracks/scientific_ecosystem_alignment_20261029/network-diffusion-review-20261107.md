# Historical deterministic network review

Recovered NetworkDiffusionModel and NetworkDiffusionInputs from 56d91dd7.
Native spillover.rs reuses Rust scalar fit/predict paths, row-normalizes weighted
adjacency, preserves isolated nodes, clips negative adjusted values and applies
historical cumulative maxima within each requested batch. Outputs are time_node;
node_time and historical_auto input orientations remain available explicitly.
Historical rate increments are sample differences, not time derivatives.

Fitting preserves independent node fitting and fixed spillover strength. Node
diagnostics are retained; aggregate diagnostics warn that adjusted predictions
need not fit observations. Summaries preserve labels, coordinates, graph
statistics and zeroed diagonals, including no-time summaries without invented
predictions. Unknown options and unsupported stochastic/intervention paths fail.
A regression caught overflow from summing finite row degrees before averaging;
the stable mean implementation now passes that case. Analytical weighted curves,
isolation, node-fit/held-out forecasts, summary provenance and invalid-shape
checks pass. Polars fitting and R/Julia weighted predictions pass locally.

Current scope supports scalar Bass, Logistic, Gompertz and Fisher-Pry bases.
Full base composition/covariates, targeted interventions, stochastic uncertainty,
scenario integration and broader network parity remain required. This family
is distinct from Kairos event/resource simulation. No shared scheduler was added.

Validation: 67 Rust tests (63 core/C ABI coverage-run plus four Polars), 29
installed Python checks, 146 Julia assertions and R native conformance. Core/C
ABI line coverage: 81.72%. Clippy, lint/types, 163-page docs, wheel/sdist metadata
and hooks pass in recorded scope. Dependencies are unchanged; the preceding audit
and its unsuppressed bincode maintenance warning remain applicable. Clean-source,
platform/hosted distribution and full scientific qualification remain open.
All 16 model and 22 feature obligations remain required. No main promotion,
publication, native tenant operation or track archival occurred.
