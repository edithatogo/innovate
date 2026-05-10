# Specification: Registry Submission Execution and Receipt Capture

## Overview

Complete the remaining external registry submissions for the project's
language packages and HPC packaging surfaces. The repository already contains
publication gates, readiness dossiers, and package sketches; this track turns
that evidence into actual registry submissions, submission receipts, or
explicit deferred states when a target cannot be completed.

The scope covers package-manager registries for Python, TypeScript, Rust, R,
Julia, Go, and C#, as well as HPC-oriented registry channels for Spack,
EasyBuild, HPSF, and E4S.

## Background

The repository documents readiness for these targets, but it does not yet
contain proof of submission. This track closes that gap by executing the
submission workflow, capturing registry responses, and updating the docs and
machine-readable status summaries to reflect the actual external state.

## Functional Requirements

1. Inventory every target registry and the package surface it owns.
2. Verify the release metadata, package contents, ownership, and credentials
   required for each submission target.
3. Submit or publish every package-manager target through its supported
   release path:
   - PyPI/TestPyPI
   - npm
   - crates.io
   - R-universe and CRAN
   - Julia General
   - Go modules
   - NuGet
4. Submit or register every HPC packaging target through its supported path:
   - Spack
   - EasyBuild
   - HPSF
   - E4S
5. Capture auditable evidence for each target, including registry URLs,
   submission receipts, build logs, and explicit blocker notes where a target
   cannot be completed.
6. Update the repository's registry-facing docs and matrices so they distinguish
   submitted, ready, blocked, deferred, and not-applicable states without
   overstating progress.
7. Preserve package name, version, and contract compatibility across all
   submission targets.

## Non-Functional Requirements

1. Submission evidence must be reproducible and easy to audit from the repo.
2. No document may imply that a registry submission succeeded without a receipt
   or public registry reference.
3. Package publication must remain gated behind the existing release and CI
   checks.
4. The track must not introduce new bindings, new package formats, or source
   tree relocations.

## Acceptance Criteria

1. Every target registry has a final status of submitted, deferred, blocked, or
   not applicable.
2. Each submitted target has a durable receipt, registry URL, or equivalent
   upstream reference in the evidence bundle.
3. The docs no longer describe submitted targets as only "ready".
4. The package publication and HPC readiness documents reflect the actual
   submission state.
5. Tests confirm the registry-status docs and evidence bundle remain in sync.
6. The track can be archived with no open submission question left unstated.

## Out of Scope

1. New community-submission campaigns for pyOpenSci, rOpenSci, JOSS, or
   NumFOCUS.
2. New source-tree reorganization.
3. New package-manager ecosystems beyond the registries listed above.
4. Scientific model research unrelated to submission execution.

