# Native innovation release gates

Current status: planned migration and evaluation. The next supported architecture is Rust-only scientific execution with Polars/Arrow and no pandas/xarray support. A policy document does not make the present hybrid source compliant.

## Alpha completion

- Scope and ownership are accurate; retired host implementations are unavailable in supported production entrypoints.
- Each retained model has explicit dynamics, observation type, parameter constraints, inference status and unsupported cases.
- Deterministic native models and interfaces pass independent references and Python-absent execution. Pandas/xarray objects are rejected with migration guidance.
- Invalid inference cannot yield posterior-labelled samples. Descriptive contrasts cannot silently become adjusted causal estimates.

## Beta

- Every advertised beta model-operation-payload combination has native dispatch, numerical, uncertainty and installed-consumer receipts.
- Clean minimal and supported-extra installs contain neither pandas nor xarray. External test-oracle environments are separate and cannot be reached by production dispatch.
- Native Polars/Arrow lifetime, null/category/unit/schema and cross-version tests pass; retained Kairos/game/Voiage integrations use accepted exact pins.
- Medicines panel and at least two non-medical datasets are qualified at the relevant semantic level; leakage-safe forecasting and failure reporting are reproducible.
- Windows x64, Linux x64 and macOS arm64 packages are tested where advertised. Other architectures/backends are explicitly unsupported until measured; no fake GPU/HPC matrix.

## Release candidate

- ABI/API/schema migration tests, FFI ownership and malformed-input fuzz/property tests pass. Resource limits, deterministic seed semantics, cancellation and failure recovery are documented.
- Rust fmt/clippy/test, Python-interface tests, meaningful mutation tests and at least 90% new-code line coverage pass without weakening stricter controls. Critical failure paths have branch tests.
- Security/dependency/licence audit, SBOM, reproducible build checks where feasible, artifact hashes and provenance are retained. Git-only dependency closures have an explicit lawful distribution route; no publishability claim from a local build.
- Independent clean-environment reproduction is recorded accurately. Automated reviewers are labelled automated; the single-maintainer policy does not invent a second human approval requirement.
- Documentation examples execute installed artefacts, not source-injected paths. Benchmarks include accuracy, failure, memory and end-to-end cost, with negative results.

## Stable scoped release

- All mandatory RC gates pass on the exact release commit and distributable artefacts; unresolved blockers prevent release.
- The support matrix enumerates stable versus experimental versus unavailable operations. Experimental research can remain outside stable scope, but every previously promised capability retains an explicit completion/retirement decision and migration record.
- Required release authorisation is obtained; publication and journal submission are separate actions, not automated consequences of a plan checkbox.
- Reproducible scientific examples and real research use support the initial software paper. Performance leadership, general empirical validity and post-frontier novelty require separate studies.

Q01–Q05 own the executable checks. The existing native/HPC alignment gates are neither silently waived nor incorrectly treated as proof of this new release.
