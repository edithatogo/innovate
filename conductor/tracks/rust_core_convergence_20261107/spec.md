# Rust-core convergence specification

## Current versus target

The current library contains bounded native deterministic slices alongside Python-owned models, payloads and subprocess bridges. The requested target is a binding-independent Rust scientific core. No current full-native claim is approved.

Extract `crates/innovate-core` for domain types, model semantics, estimation and diagnostics; `crates/innovate-arrow` for interchange; and optional `crates/innovate-kairos` for simulation integration. Keep direct Python/PyO3 and versioned native FFI adapters under `bindings/`. These are proposed paths, not existing delivered crates.

Python owns convenience, notebooks and plotting, not a duplicate production implementation of stable native mathematics. Explicit external inference backends are allowed but must identify their runtime and cannot count towards native completion. Existing APIs receive a versioned migration path rather than silent removal.

## Native acceptance

Every stable native model-operation-payload combination executes with Python absent. Python interfaces dispatch to the same Rust core used by native consumers. Unsupported combinations produce structured errors; no undisclosed Python subprocess fallback. Test actual dispatch, not only matching output.

Use typed counts/stocks/rates/shares, units, time grids, observation likelihoods, constraints and inference states. Arrow is the interchange contract; internal numerical representations may differ. Kairos dependencies are optional so basic deterministic forecasting does not build an entire simulation stack.

## Dependency strategy

Inspect `repository-standards` and Voiage's current native interfaces before implementation. Qualify maintained candidates such as argmin, Diffsol and nuts-rs against required algorithms, licensing, maintenance, packaging and numerical oracles. Do not adopt a candidate or change a pin on name alone. Preserve Python/R references as test-oracle dependencies, not hidden native runtime dependencies.

## Scope control

Prioritise a verified Python/Rust stable surface, retaining supported binding commitments through compatibility tests. GPU, HPC and further package-manager work remain evidence-gated follow-ons, not prerequisites for every numerical correction. The existing alignment track retains responsibility for accepted upstream pins and unfinished external/native qualification.
