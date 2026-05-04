# Implementation Plan: R Package PDF Manual and Vignette Publication Quality Gate

## Phase 1: Standards Audit and Red-Phase Gates

- [x] Task: Audit R manual and vignette publication requirements
    - [x] Confirm CRAN/R-universe expectations for reference manuals, vignettes, examples, and generated artifacts
    - [x] Inventory current R package documentation, exports, examples, and workflow coverage
    - [x] Decide whether to continue handwritten Rd docs or introduce roxygen2, with rationale in track notes
- [x] Task: Add failing quality gates for the missing manual path
    - [x] Add a static test that detects skipped R manual generation in release-critical workflows
    - [x] Add a static test that requires a documented manual generation command and artifact policy
    - [x] Add a static test that checks exported R symbols have matching Rd aliases
- [x] Task: Define six-subagent work ownership
    - [x] Record disjoint write scopes for reference docs, vignettes, CI, metadata, tests, and release docs
    - [x] List validation commands each subagent can run independently
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Standards Audit and Red-Phase Gates' (Protocol in workflow.md)

## Phase 2: R Reference Manual Completeness

- [x] Task: Complete Rd reference documentation
    - [x] Add or update package-level documentation for `innovate.R`
    - [x] Ensure every exported function has aliases, usage, arguments, return values, details, examples, and seealso links where appropriate
    - [x] Keep examples deterministic and safe for `R CMD check`
- [x] Task: Add local manual build command support
    - [x] Document the exact `R CMD Rd2pdf` or equivalent command for local manual builds
    - [x] Add a repository script or make target only if it matches existing project conventions
    - [x] Verify the generated manual name includes package identity and version metadata where feasible
- [x] Task: Validate manual generation
    - [x] Run R documentation checks without suppressing manual generation where local tooling permits
    - [x] Record any local TeX dependency constraints and CI fallback behavior
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 2: R Reference Manual Completeness' (Protocol in workflow.md)

## Phase 3: Vignette and User-Facing Package Documentation

- [x] Task: Add a buildable R package vignette
    - [x] Create an end-to-end vignette under `bindings/r/vignettes/`
    - [x] Cover installation, backend discovery, core examples, diagnostics, and failure modes without network-dependent execution
    - [x] Use guarded examples for optional Python/Rust bridge behavior
- [x] Task: Wire vignette metadata
    - [x] Add required `Suggests` and `VignetteBuilder` fields to `bindings/r/DESCRIPTION`
    - [x] Update `bindings/r/.Rbuildignore` to avoid excluding required vignette sources or generated outputs incorrectly
    - [x] Confirm the package builds without committing generated vignette artifacts unless the release policy requires them
- [x] Task: Validate vignette behavior
    - [x] Run vignette build/check commands where dependencies are available
    - [x] Add or update tests that assert vignette metadata and source files are present
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Vignette and User-Facing Package Documentation' (Protocol in workflow.md)

## Phase 4: CI/CD and Publication Integration

- [x] Task: Add CI manual PDF artifact gate
    - [x] Configure R and LaTeX/TinyTeX setup for the R manual job
    - [x] Build the R PDF manual in CI and upload it as an artifact
    - [x] Make artifact names stable and include package/version context where practical
- [x] Task: Harden R publication workflow
    - [x] Ensure release-critical R checks build or explicitly validate the manual
    - [x] Remove `--no-manual` from release-critical paths unless a separate mandatory manual job covers the same release
    - [x] Keep fast checks separate from release gates when runtime or TeX setup cost requires it
- [x] Task: Add CI policy tests
    - [x] Assert workflow coverage for TeX setup, manual generation, artifact upload, and publication blocking
    - [x] Validate that retained `--no-manual` usage is documented and paired with a mandatory manual job
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 4: CI/CD and Publication Integration' (Protocol in workflow.md)

## Phase 5: Release Documentation and Final Review

- [x] Task: Update user and maintainer documentation
    - [x] Update `bindings/r/README.md` with manual and vignette build instructions
    - [x] Update `bindings/r/cran-comments.md` with documentation validation notes
    - [x] Update binding publication docs with R PDF artifact behavior
- [x] Task: Run full validation
    - [x] Run targeted repository tests for R documentation policy
    - [x] Run R package checks and manual/vignette build commands where local tooling permits
    - [x] Run workflow/static validation such as action linting where configured
- [x] Task: Final track review and archival readiness
    - [x] Run final `conductor-review` across the whole track
    - [x] Apply high-confidence fixes, rerun validation, and repeat up to two review loops
    - [x] Prepare the track for archive once all phases are complete and CI gates pass
- [x] Task: Conductor - Automated Review and Checkpoint 'Phase 5: Release Documentation and Final Review' (Protocol in workflow.md)
