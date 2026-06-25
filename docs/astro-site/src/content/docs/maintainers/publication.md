---
title: Publication
description: Binding publication and release gate guidance.
---

# Publication

The publication gate remains release-driven now that Astro/Starlight is the
active docs site.

Current publication targets:

- PyPI/TestPyPI for Python
- npm for TypeScript package `innovate.ts`
- crates.io for Rust package `innovate-rs`, exposed to users as `innovate.rs`
- R-universe and CRAN for R package `innovate.R`
- Julia General for Julia package `Innovate`, exposed to users as `innovate.jl`
- Go modules for `innovate.go`
- NuGet for C# package `innovate.cs`

The site migration does not change the publication policy; it only changes the
docs surface that explains it.

Publication status and external handoff state are recorded in
`docs/source/_static/registry_submission_receipts.json`. Those receipts are the
source for current registry URLs, deferred targets, and maintainer-managed
handoff states. These maintainer-managed handoff states separate local package
readiness from external registry review and credential-controlled publication.

## Required package gates

- Python: `uv build` and `twine check dist/*`.
- TypeScript: `npm run schema:check`, `npm run typecheck`, `npm test`, and `npm pack --dry-run`.
- Rust: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`, and `cargo package`.
- R: binding tests, `R CMD build`, `R CMD check --as-cran`, and maintainer PDF manual inspection.
- Julia: `Pkg.instantiate()`, `Pkg.test()`, compatibility bounds for `JSON` and `julia`, and installed-package smoke validation.
- Go: `go test ./...` plus the `bindings/go/v0.5.0` submodule tag pattern.
- C#: `dotnet test` and `dotnet pack` on .NET 10 and .NET 11 with bridge-content inclusion.

## Binding package metadata

The package gates require aligned version `0.5.0` across Python and every
binding. The NuGet package must include the bridge file at
`contentFiles/any/any/innovate/kernel_bridge.py`; this bridge-content check
protects thin binding installation behavior.

Julia registry readiness includes installed-package smoke validation from a
copied package tree. NuGet readiness includes package metadata, symbol package
generation, and validation that the runtime bridge asset is present.

## R publication artifacts

The R binding includes a source vignette under `bindings/r/vignettes/`.
Maintainers should generate the PDF manual with
`R CMD Rd2pdf bindings/r --output=innovate.R-manual.pdf` before R-universe or
CRAN publication. The release workflow uploads the manual artifact as
`r-manual-${{ steps.r_metadata.outputs.package }}-${{ steps.r_metadata.outputs.version }}`.
Generated source tarballs, `.Rcheck` directories, PDF manuals, and built
vignette outputs are inspection artifacts and are not committed.
