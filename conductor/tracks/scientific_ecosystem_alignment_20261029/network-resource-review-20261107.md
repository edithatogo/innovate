# Complementary goods and network/resource review

Same isolated writing lane, based on 9a721b9. Complementary-goods equations and
parameter ordering were recovered from the preserved source at 56d91dd7. The
native trajectory engine provides fit, prediction, signed rates, summary and
diagnostics. Independent SciPy DOP853 fixtures, decoupled and symmetric analytic
limits, perturbed-start parameter recovery and all three installed binding
consumers validate the bounded port. Positive interactions can produce levels
above one; no clipping or simplex normalization is imposed.

Kairos ABMContext owns node identities/components and dispatch, and its Resource
owns FIFO leases/queueing. Susceptible nodes have hazard p plus q times adopted
neighbor fraction. Pending agents cannot request twice or transmit. Adoption
occurs on service completion; tied completions precede requests. Node positions
form a static undirected unweighted graph. Request/completion traces directly
verify FIFO order and one-server versus three-server timing. Tests also cover
pure contagion, isolation, replay, capacity bounds, pending-state conservation,
invalid graphs and singleton R initial-node transport.

The simulation p domain now permits zero innovation; deterministic Bass retains
its previous positive-p domain. Network forcing/covariates are explicitly
rejected. Dynamic/directed/weighted networks, heterogeneous node hazards,
resource priorities/abandonment, resume and general network_diffusion model
parity remain required. RNG checkpoints are provenance only. Complementary-goods
covariates/events and extreme-interaction numerical qualification remain open.

Validation: 57 Rust tests (53 core/C ABI coverage-run, 4 Polars), 27 installed
Python checks, 142 Julia assertions and R native conformance. Core/C ABI line
coverage: 81.26%; network module: 91.39%. Clippy, lint/types, docs and package
metadata pass. Dependency audit finds no vulnerabilities and retains the existing
unsuppressed bincode maintenance warning. Clean-source, platform/hosted package
and full scientific qualification remain open. No main promotion or publication
occurred; all 16 model and 22 feature obligations remain required.
