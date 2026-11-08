# Architecture Modernization Roadmap

The current direction is [ADR 0006](adr/0006-rust-polars-native-only.md): Rust owns
execution, Polars owns tables, and only Python/Julia/R bindings are supported.
The 0.7 development cutover permits interim unavailability but retains all
scientific features as obligations. Earlier “completed” track labels do not
constitute native or scientific acceptance.

Use [the active migration plan](../conductor/tracks/scientific_ecosystem_alignment_20261029/migration-plan.md)
and [feature ledger](../migration-ledger.json) for current scope and status.
The complete pre-migration roadmap remains at the inventory's source commit in
Git history, including earlier accelerator, inference, reporting, remote-runtime
and community/HPC intentions. None are silently removed from final scope.

Priorities are: native scientific correctness and error contracts; remaining
models and inference; Kairos simulation dispatch; Polars/Arrow interoperability;
installed R/Julia native distribution; platform and scientific qualification.
Recommendations include identifiability, calibrated uncertainty, time-aware
forecast evaluation, explicit incident/cumulative semantics, paired seeded
scenario comparison and reproducible provenance bundles.
