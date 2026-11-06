# ADR 0006: Rust-owned scientific core and Polars

Accepted by user direction on 6 November 2026; implementation remains in progress.
This supersedes the runtime, dataframe and language-surface choices in ADRs
0002–0004, while preserving their historical evidence.

Rust owns scientific state, validation, computation and diagnostic semantics.
Python, Julia and R are thin supported bindings. TypeScript, Go and C# are removed.
Polars owns tabular processing; Arrow IPC/Parquet provides durable interchange.
The Python distribution uses PyO3/Maturin and has Polars as its sole base dependency.
No pandas/xarray, SciPy/NumPy numerical runtime or Python subprocess fallback is
part of the new package. External comparison consumers may have their own
isolated dependencies; they do not become Innovate runtime dependencies.

The user permits breaking API refactoring and interim unavailability but requires
all current and planned features in the final Rust library. Each missing area
remains open in `migration-ledger.json`. Development tests and a built wheel do
not close full scientific, multi-platform, packaging or release acceptance.

Voiage alignment: separated domain/numerics/serialization/binding responsibilities,
Polars-backed tabular contracts, strict identities and installed consumer tests.
Kairos alignment: reuse upstream scheduling, DES/ABM state and seeded replay;
do not implement a competing scheduler. Do not inherit additional languages,
optional GPU/distributed services or experimental dependency pins by analogy.

The schema changes to 2.0 because Gompertz semantics, error handling and supported
surfaces change. Capability discovery advertises implemented bounded paths only.
Scientific parity includes analytic oracles and parameter recovery, not just
matching a potentially faulty former implementation. Uncomputed diagnostics
are explicit; uncertainty and identifiability require their own implementation.
