# Specification: External Submission Blocker Closure

## Overview

Previous tracks prepared registry and HPC submission evidence, but several
targets remain blocked or maintainer-managed. This track converts those states
into final outcomes: submitted with receipt, blocked with current external
reason, deferred with owner and revisit condition, or removed from the roadmap
claim surface.

## Functional Requirements

1. Audit registry submission receipts, community readiness matrices, HPC
   submission packets, and governance notes.
2. For each target, record final state: submitted, blocked, deferred,
   ready-for-maintainer, or not-applicable.
3. Add or update evidence artifacts for CRAN, Julia General, NuGet, npm,
   crates.io, Go modules, R-universe, Spack, EasyBuild, HPSF, E4S, pyOpenSci,
   rOpenSci, JOSS, NumFOCUS, PyPA, Apache Arrow, and .NET Foundation as
   applicable.
4. Add tests that reject unowned blocked states and overclaimed submission
   success.
5. Update roadmap and docs to distinguish readiness from submission.

## Non-Functional Requirements

1. Do not claim external acceptance without a receipt or durable public link.
2. Do not require credentials or maintainer-only actions to pass local tests;
   record those as explicit blocked or ready-for-maintainer states.
3. Commit after every task and run `conductor-review` after every phase and full
   track completion.

## Acceptance Criteria

1. Every external target has a current status, owner, evidence link, and next
   action or closure rationale.
2. Tests fail if a target remains generically blocked without current evidence.
3. No docs page implies submitted or accepted status where only readiness exists.
4. HPC and community submission packets are regenerated from source artifacts.

## Out of Scope

1. Publishing to external registries without maintainer credentials.
2. Changing package names or versioning policy.
3. Implementing Rust-native model slices.
