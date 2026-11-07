# Dependency alignment and qualification — 7 November 2026

Status: inspected candidates and implementation rules; no dependency lock is changed or qualified by this document.

| Layer | Observed candidate/baseline | Action |
|---|---|---|
| Rust toolchain | Rust 1.99.1; Kairos declares Rust 1.99 | Begin with 1.99.1, inspect newer candidates at execution, pin exact toolchain and re-run full supported native consumers. |
| Python interface/tooling | Python 3.15.2 is the current maintenance/security candidate and Kairos tooling direction; Innovate/Voiage declarations still include 3.15.1 | Qualify 3.15.2 for bindings/tooling; Python is not the scientific engine. Do not rewrite historical 3.15.1 receipts. |
| Rust Polars closure | Kairos `kairo-ecs-arrow` uses `pola-rs/polars` revision `96439685eb2f95108f51087d2b76091917f2593c`, declared 0.55.1, and matching `polars-arrow` | Start with the exact shared closure; inspect released/newer alternatives rather than assume version labels imply compatibility. |
| Required root patch for that closure | `pola-rs/hashbrown` revision `d76a6f81abd8f388676aff15068236a8d21c7e1a` | Reproduce and qualify the patch in the consuming workspace. Do not omit it or publish an unpublishable Git closure as a crates.io package. |
| Native Arrow logical schema | Kairos optional `arrow-schema` 60.1.0 | Test schema/buffer/lifetime interchange with Polars; do not equate Arrow Rust and PyArrow major numbers. |
| Python tabular boundary | Voiage uses Polars >=2.1,<3 and PyArrow >=26,<27 | Qualify latest exact compatible candidates through installed native round trips. No pandas/xarray bridge. |
| Bindings/build | PyO3/maturin; current PyO3 candidate 0.30.1 | Verify CPython ABI, free-threading claims, wheel platforms and failure paths. No blanket ABI claim. |
| Numerical building blocks | argmin, Diffsol, nuts-rs; existing ecosystem Rayon, rand, Criterion, proptest and Clarabel where relevant | Review algorithm, maintenance, licence, API and resource cost; use maintained implementations when they remove custom code and pass independent oracles. |

Official candidate sources: https://blog.rust-lang.org/2026/10/15/Rust-1.99.1/ ; https://www.python.org/downloads/release/python-3152/ ; https://github.com/pola-rs/polars/releases ; https://pyo3.rs/ ; https://docs.rs/nuts-rs/ ; https://docs.rs/argmin/ ; https://martinjrobins.github.io/diffsol/ . Exact candidate metadata is refreshed and recorded at execution; this table is not an evergreen latest-version assertion.

## Binding decisions

Rust core dependencies are separate from Polars/data, optional simulator/game adapters, bindings, build tools, external test oracles and documentation. General-purpose scientific Python packages must not re-enter the supported production closure through diagnostics or plotting extras. In particular, a convenient ArviZ/xarray or pandas-based plotting dependency is not an exception to the new support policy; use native/Polars diagnostics and boundary arrays, or isolate external oracle workflows.

The mandatory gate includes both source/API scans and installation checks: no pandas/xarray requirement or import in shipped production paths; explicit rejection of those objects; clean supported-extra resolution without their installation; missing native library fails rather than executing Python mathematics. Tests may use isolated external references with a separate lock/environment.

## Promotion and packaging

For every candidate retain exact source/lock hashes and security, build, numerical, installed-consumer and resource receipts. All five must pass before promotion. A retained older dependency requires capability-specific blocker, owner and dated revisit condition. Replay-changing RNG, solver tolerances and schema changes require an explicit algorithm/schema version and consumer requalification.

Separate development-source qualification from distributable package qualification. The current Kairos Git-Polars crate is marked `publish=false`; a stable public package needs an allowed immutable distribution route or a separately qualified released dependency closure. Never remove that gate simply to publish.
