# Specification: R Package PDF Manual and Vignette Publication Quality Gate

## Problem

R packages commonly ship with a LaTeX-rendered reference manual and, when appropriate, vignettes that are built during package checks and publication workflows. The current R binding package has Rd documentation, but the repository does not provide a generated or CI-validated PDF manual. The existing CI and publication workflows run `R CMD check --as-cran --no-manual`, which explicitly skips the manual build. That leaves a gap for CRAN-style documentation quality, release artifacts, and user-facing package documentation.

## Goals

- Make the R binding package documentation meet modern CRAN/R-universe expectations for reference manuals, examples, package metadata, and optional vignettes.
- Add a reproducible PDF manual generation path using R's standard tooling, such as `R CMD Rd2pdf` and/or `R CMD check --as-cran` without suppressing manuals in the manual validation path.
- Add CI gates that install the required LaTeX toolchain in a controlled way, build the R reference manual, and upload the generated PDF as an artifact.
- Add or update R package docs so every exported API has complete Rd coverage, examples, aliases, argument descriptions, return values, and package-level context.
- Add a user-facing vignette if it improves the package's publication quality and can be built without fragile network or environment dependencies.
- Document the release behavior clearly: generated PDFs are CI/release artifacts unless a deliberate policy says otherwise.
- Keep the work easy to parallelize across six subagents with disjoint file ownership and clear validation responsibilities.
- Automatically run `conductor-review` at the end of every phase and at track completion, apply high-confidence fixes, rerun validation, and progress to the next phase without manual intervention.

## Non-Goals

- Publishing to CRAN or R-universe during this track.
- Rewriting the R binding API unless documentation work exposes a correctness issue.
- Committing generated build products such as `inst/doc` or PDF artifacts unless the release policy explicitly requires it.
- Replacing all handwritten Rd files with roxygen2 unless the audit proves that the migration materially improves maintainability.

## Current Evidence

- `bindings/r/man/kernel_bridge.Rd` exists.
- `bindings/r/vignettes/` and `bindings/r/inst/doc/` are absent.
- `.github/workflows/ci.yml` and `.github/workflows/bindings-publish.yml` use `R CMD check --as-cran --no-manual`.
- `bindings/r/DESCRIPTION` does not currently declare vignette tooling such as `knitr`, `rmarkdown`, or `VignetteBuilder`.

## SOTA Criteria

- Reference manual generation is tested in CI with an explicit TeX setup, preferably through maintained R GitHub Actions such as `r-lib/actions/setup-r`, `setup-r-dependencies`, and a TinyTeX or equivalent LaTeX setup.
- The regular fast CI path may keep a lightweight check only if a separate mandatory manual/PDF job covers the skipped surface.
- The publication workflow blocks release when the R manual cannot be generated.
- Generated PDF artifacts are uploaded from CI and release workflows with stable names that include package and version metadata.
- Rd documentation coverage is machine-checked for every exported symbol in `NAMESPACE`.
- Examples are CRAN-safe, deterministic, and guarded when they require an optional Python/Rust backend or local runtime state.
- Vignettes build under `R CMD build`/`R CMD check` without network calls or long-running examples.
- Documentation policy is captured in package README, CRAN comments, and binding publication docs.

## Reference Sources

- R Core, "Writing R Extensions" (`https://cran.r-project.org/doc/manuals/r-devel/R-exts.html`): `R CMD Rd2pdf` generates PDF output from Rd files and can operate on a package source directory.
- R Core, "Writing R Extensions" (`https://cran.r-project.org/doc/manuals/r-devel/R-exts.html`): package vignettes are part of package source/build behavior unless explicitly suppressed by build options.
- `r-lib/actions` (`https://github.com/r-lib/actions`): maintained GitHub Actions include R setup, package checking, dependency setup, and TinyTeX setup for LaTeX-backed documentation jobs.

## Parallelization Plan

This track is designed for six subagents with non-overlapping ownership:

- **Agent A: R Reference Docs** owns `bindings/r/man/`, package-level Rd docs, examples, aliases, and exported symbol documentation coverage.
- **Agent B: Vignette and User Guide** owns `bindings/r/vignettes/`, vignette build metadata, and examples that can run under package checks.
- **Agent C: CI Manual Artifact Gate** owns `.github/workflows/ci.yml`, `.github/workflows/bindings-publish.yml`, and any workflow artifact naming or TeX setup.
- **Agent D: Package Metadata and CRAN Policy** owns `bindings/r/DESCRIPTION`, `bindings/r/.Rbuildignore`, `bindings/r/cran-comments.md`, and CRAN/R-universe compliance notes.
- **Agent E: Tests and Static Quality Gates** owns R documentation tests and Python repository tests that assert manual/vignette workflow coverage.
- **Agent F: Release Documentation and Review** owns `bindings/r/README.md`, `docs/source/`, release notes, final audit checklists, and conductor-review remediation tracking.

## Acceptance Criteria

- The repository documents that the R PDF manual was previously missing and now has an implementation track.
- A CI job builds the R PDF manual with a real LaTeX toolchain and uploads it as an artifact.
- The R publication workflow validates manual generation before publishing or preparing release artifacts.
- `--no-manual` is removed from release-critical R checks, or a separate mandatory manual generation job justifies any retained lightweight usage.
- Every exported R symbol has Rd documentation and coverage is checked by tests or CI.
- A package-level manual page and, where appropriate, a vignette are present and buildable.
- CRAN-safe examples avoid network access, long runtime, and unguarded optional backend assumptions.
- R package metadata declares any vignette or documentation build dependencies correctly.
- Package README, CRAN comments, and binding publication docs explain how to build and retrieve the PDF manual.
- Each phase ends with the Conductor automated review/checkpoint task, and implementation must invoke `conductor-review`, apply fixes, rerun tests, and auto-progress.
