---
title: Polyglot Registry Plan
description: Registry strategy for package and scientific submissions.
---

# Polyglot Registry Plan

This document records the registry plan for the polyglot `innovate` surface. It
separates package-manager publication, scientific-community submissions, and
HPC-oriented registry readiness so the repository can track each path
independently.

The plan is intentionally conservative: the repository contains readiness
evidence and package gates. This page does not claim that any external submission has already been completed.
Use the readiness dossiers as submission checklists, not as proof of submission.

## Registry Layers

**Package-manager registries** are the language-specific publication targets for
the bindings and core package surfaces: PyPI/TestPyPI, npm, crates.io,
R-universe/CRAN, Julia General, Go modules, and NuGet.

**Scientific community submissions** are the reviewer-facing targets for
pyOpenSci, rOpenSci, JOSS, NumFOCUS, Apache Arrow, Julia community, R community,
and Julia General registry readiness.

**HPC registries** are the packaging and registry targets for Spack, EasyBuild,
HPSF, and E4S.

## Recommended Sequence

1. Keep the package-manager publication gates aligned with the binding CI jobs.
2. Keep the scientific submission dossiers aligned with the readiness matrices.
3. Keep the HPC contract aligned with package sketches and scheduler evidence.
4. Treat external registry submission as a release decision, not a doc-only milestone.

## Current Status By Layer

| Layer | Current status | Next action |
| --- | --- | --- |
| Package-manager registries | Submitted, deferred, or ready-for-review by target | Use `docs/source/_static/registry_submission_receipts.json` and `docs/source/_static/external_submission_target_inventory.json` as the source of truth before any release claim. |
| Scientific community submissions | Ready-for-review or not-applicable by target | Use the readiness dossiers as submission checklists, not as proof of submission. |
| HPC registries | Spack/EasyBuild ready_for_review; HPSF/E4S ready_for_maintainer | Use the HPC registry contract, scheduler evidence, package sketches, and closure inventory before any upstream registry claim. |

## Planned Registry Artifacts

The plan relies on the following repository artifacts:

- binding publication CI evidence in
  `docs/astro-site/src/content/docs/maintainers/publication.md`;
- community submission evidence in
  `docs/astro-site/src/content/docs/operations/community-readiness.md`;
- HPC registry evidence in
  `docs/astro-site/src/content/docs/operations/hpc-readiness.md`;
- governance and sustainability evidence in
  `docs/astro-site/src/content/docs/operations/governance.md`;
- polyglot architecture guidance in
  `docs/astro-site/src/content/docs/architecture/polyglot-repo.md`;
- the HPC registry contract in
  `docs/astro-site/src/content/docs/operations/hpc-registry.md`;
- target-level closure state in
  `docs/source/_static/external_submission_target_inventory.json`.

## Non-Goals

This plan does not claim that any registry submission has been made, and it does
not introduce new package formats or language bindings. It only records the path
from readiness evidence to submission.

Use the HPC registry contract to keep the HPC registry path separate from
package-manager publication.

Canonical source:

- `docs/astro-site/src/content/docs/operations/polyglot-registry.md`
