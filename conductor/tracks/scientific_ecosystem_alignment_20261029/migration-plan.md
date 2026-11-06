# Rust and Polars migration

User-approved scope, 6 November 2026. This is a breaking development line;
interim gaps are allowed, final loss of existing/planned features is not.

- [x] Inspect exact upstream Voiage/Kairos sources and official dependency registries.
- [x] Preserve source/test feature inventory with hashes and the pre-migration commit.
- [x] Extract Rust-only kernel; remove subprocess fallbacks; add native regression tests.
- [x] Make Polars the data contract and install the native Python extension (bounded macOS consumer proof).
- [x] Replace Julia/R Python launchers with native library bindings and consumer tests (local C ABI; distribution remains open).
- [x] Remove TypeScript/Go/C# packages and active support claims/jobs.
- [ ] Port remaining model families, inference, diagnostics, preprocessing, scenarios,
      data integration, benchmarking, reporting, plugins and remote-execution contracts.
- [x] Qualify selected installed Voiage 3.1.0 Polars/Arrow CPU handoff against an analytic EVPI oracle.
- [ ] Qualify current Kairos native dispatch and full scenario/uncertainty interoperability.
- [~] Reconcile active dependencies, documentation, CI and packaging; local native gates pass, full scientific/platform release gates remain open.
- [ ] Integrator review and acceptance. No release or hosted acceptance is implied.

## Architecture

Rust owns scientific validation, state, numeric computation, diagnostics and
versioned capabilities. Polars DataFrames/LazyFrames own tabular data; Arrow
IPC/Parquet is the durable language-neutral exchange. Bindings only adapt inputs,
invoke the core, and format results. No numerical Python fallback is permitted.
Unavailable capabilities return an explicit typed error and remain in the
feature ledger; a passing subset never closes the full migration.

Voiage owns VOI/economic analysis; reuse its typed, Rust-owned boundaries and
Polars/Arrow contracts, not its economic kernels. Kairos owns virtual time,
DES/ABM scheduling, deterministic streams and replay; do not copy its scheduler.
Sources inspected: Voiage a5d87a736fa9f0f876e44872d946b2170d2a0898,
Kairos d56863505d938feb069af88a737eb7668113d14b. Their local main checkouts were
read only and were not synchronised, reset or otherwise altered.

## Feature preservation

`feature-source-inventory-20261106.json` retains pre-migration paths, hashes,
public Python definitions and source commit. The model registry has 16 named
families, while the existing Rust code implements bounded payloads for only
Bass, Logistic, Gompertz, Fisher-Pry and Norton-Bass. Python-only functionality
and roadmap proposals are still required, regardless of earlier archived labels.
The inventory is an implementation input, not evidence of native equivalence.

## Recommended additions

Prioritise identifiability and fit-convergence diagnostics; time-aware held-out
forecast scoring and interval calibration; explicit cumulative-versus-incident
adoption semantics; scenario comparators with paired deterministic seeds;
missing-data/irregular-time policies; provenance and reproducible Arrow bundles.
These strengthen scientific use rather than adding more language surfaces.
GPU/distributed execution requires measured benefit and deterministic contracts
before promotion; it is not a prerequisite for honest CPU capability reporting.

## Native cutover checkpoint — 7 November 2026

See [qualified local evidence](native-cutover-20261107.json) and
[review and remaining gates](native-cutover-review-20261107.md). Source, deletion
and artifact hashes make this a reviewable isolated proposal. Earlier evidence
and assignment history remain intact; no prior Python suite is counted as passing.

The next unblocked scientific work is native multi-generation Norton–Bass and
fit-convergence/identifiability reporting, starting from the retained source/test
inventory. The ecosystem lane can then qualify the current Kairos adapter and
scenario contract without copying scheduler internals. Continue the 38 required
model/feature obligations; a local five-model pass does not close them.
