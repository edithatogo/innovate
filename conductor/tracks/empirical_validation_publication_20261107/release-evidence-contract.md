# Scientific release evidence contract

The release checker validates the evidence after every migration obligation is
declared qualified. The ledger's `qualified_source_sha256` must match the clean
Git source inventory. Set `release_evidence.manifest` and
`release_evidence.attestation` to relative paths for the qualification manifest
and its GitHub attestation bundle. The qualification source commit must be
available locally and have the same source inventory as the release checkout.
This allows the evidence to be recorded after execution without changing the
qualified source.

## Manifest and receipts

The manifest uses `schema_version: 1`, `kind: innovate-release-qualification`,
the full `source_commit` and its `source_sha256`. Its `obligations` object has
exactly one entry for each required model and feature. Each entry references a
`native` and a `scientific` report. The scientific reference agrees with the
ledger's evidence path and digest.

The `installed_consumers` object covers `linux-x86_64`, `macos-arm64` and
`windows-x86_64`, with an `artifact` and a `report` for each. References
contain a relative `path` and the file's `sha256`. An installed report records
the tested artifact's `artifact_sha256`.

Reports use `schema_version: 1`, the same `source_sha256`, and a kind of
`native_execution`, `scientific_acceptance` or `installed_execution`. Each
successful report has `status: passed` and `results` keyed by obligation or
platform. Each result records counts for `passed`, `failed`, `skipped` and
`errors`. At least one check must have passed in each claimed scope. The three
other counts must be zero. The producer is responsible for running all required
cases and recording the accepted scientific scope from the track.

## Verification and current scope

The verifier binds the exact manifest bytes to a GitHub SLSA provenance
attestation. The repository and signer are fixed to Innovate and its
`release-readiness.yml` workflow. The source and signer commits must both
match the qualification commit. GitHub CLI verifies the bundle. [GitHub attestation verification](https://cli.github.com/manual/gh_attestation_verify)

The current ledger still has 38 incomplete obligations. The existing dry-run
job produces packaging artifacts. A real producer run with complete native,
scientific and installed evidence still needs to supply the signed bundle.
Synthetic reports exercise the verifier and provide no release evidence.
