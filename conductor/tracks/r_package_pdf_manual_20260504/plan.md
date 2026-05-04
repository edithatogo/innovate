# Implementation Plan: R Package PDF Manual and Vignette Publication Quality Gate

## Phase 1: Standards Audit and Red-Phase Gates

- [ ] Task: Audit R manual and vignette publication requirements
    - [ ] Confirm CRAN/R-universe expectations for reference manuals, vignettes, examples, and generated artifacts
    - [ ] Inventory current R package documentation, exports, examples, and workflow coverage
    - [ ] Decide whether to continue handwritten Rd docs or introduce roxygen2, with rationale in track notes
- [ ] Task: Add failing quality gates for the missing manual path
    - [ ] Add a static test that detects skipped R manual generation in release-critical workflows
    - [ ] Add a static test that requires a documented manual generation command and artifact policy
    - [ ] Add a static test that checks exported R symbols have matching Rd aliases
- [ ] Task: Define six-subagent work ownership
    - [ ] Record disjoint write scopes for reference docs, vignettes, CI, metadata, tests, and release docs
    - [ ] List validation commands each subagent can run independently
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 1: Standards Audit and Red-Phase Gates' (Protocol in workflow.md)

## Phase 2: R Reference Manual Completeness

- [ ] Task: Complete Rd reference documentation
    - [ ] Add or update package-level documentation for `innovate.R`
    - [ ] Ensure every exported function has aliases, usage, arguments, return values, details, examples, and seealso links where appropriate
    - [ ] Keep examples deterministic and safe for `R CMD check`
- [ ] Task: Add local manual build command support
    - [ ] Document the exact `R CMD Rd2pdf` or equivalent command for local manual builds
    - [ ] Add a repository script or make target only if it matches existing project conventions
    - [ ] Verify the generated manual name includes package identity and version metadata where feasible
- [ ] Task: Validate manual generation
    - [ ] Run R documentation checks without suppressing manual generation where local tooling permits
    - [ ] Record any local TeX dependency constraints and CI fallback behavior
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 2: R Reference Manual Completeness' (Protocol in workflow.md)

## Phase 3: Vignette and User-Facing Package Documentation

- [ ] Task: Add a buildable R package vignette
    - [ ] Create an end-to-end vignette under `bindings/r/vignettes/`
    - [ ] Cover installation, backend discovery, core examples, diagnostics, and failure modes without network-dependent execution
    - [ ] Use guarded examples for optional Python/Rust bridge behavior
- [ ] Task: Wire vignette metadata
    - [ ] Add required `Suggests` and `VignetteBuilder` fields to `bindings/r/DESCRIPTION`
    - [ ] Update `bindings/r/.Rbuildignore` to avoid excluding required vignette sources or generated outputs incorrectly
    - [ ] Confirm the package builds without committing generated vignette artifacts unless the release policy requires them
- [ ] Task: Validate vignette behavior
    - [ ] Run vignette build/check commands where dependencies are available
    - [ ] Add or update tests that assert vignette metadata and source files are present
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 3: Vignette and User-Facing Package Documentation' (Protocol in workflow.md)

## Phase 4: CI/CD and Publication Integration

- [ ] Task: Add CI manual PDF artifact gate
    - [ ] Configure R and LaTeX/TinyTeX setup for the R manual job
    - [ ] Build the R PDF manual in CI and upload it as an artifact
    - [ ] Make artifact names stable and include package/version context where practical
- [ ] Task: Harden R publication workflow
    - [ ] Ensure release-critical R checks build or explicitly validate the manual
    - [ ] Remove `--no-manual` from release-critical paths unless a separate mandatory manual job covers the same release
    - [ ] Keep fast checks separate from release gates when runtime or TeX setup cost requires it
- [ ] Task: Add CI policy tests
    - [ ] Assert workflow coverage for TeX setup, manual generation, artifact upload, and publication blocking
    - [ ] Validate that retained `--no-manual` usage is documented and paired with a mandatory manual job
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 4: CI/CD and Publication Integration' (Protocol in workflow.md)

## Phase 5: Release Documentation and Final Review

- [ ] Task: Update user and maintainer documentation
    - [ ] Update `bindings/r/README.md` with manual and vignette build instructions
    - [ ] Update `bindings/r/cran-comments.md` with documentation validation notes
    - [ ] Update binding publication docs with R PDF artifact behavior
- [ ] Task: Run full validation
    - [ ] Run targeted repository tests for R documentation policy
    - [ ] Run R package checks and manual/vignette build commands where local tooling permits
    - [ ] Run workflow/static validation such as action linting where configured
- [ ] Task: Final track review and archival readiness
    - [ ] Run final `conductor-review` across the whole track
    - [ ] Apply high-confidence fixes, rerun validation, and repeat up to two review loops
    - [ ] Prepare the track for archive once all phases are complete and CI gates pass
- [ ] Task: Conductor - Automated Review and Checkpoint 'Phase 5: Release Documentation and Final Review' (Protocol in workflow.md)
