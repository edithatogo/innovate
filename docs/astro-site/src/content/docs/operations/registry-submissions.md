---
title: Registry Submission Receipts
description: Current evidence bundle for binding and HPC registry targets.
---

# Registry Submission Receipts

The receipts page records live links and explicitly scoped pending targets.

Submitted targets:

- PyPI/TestPyPI
- npm
- crates.io
- R-universe
- Julia General trigger
- Go modules
- NuGet

Deferred, review-ready, or blocked targets:

- CRAN: deferred until maintainer send.
- Spack: ready_for_review, pending maintainer-managed upstream submission.
- EasyBuild: ready_for_review, pending maintainer-managed upstream submission.
- HPSF: blocked until a governance submission endpoint or review contact is identified.
- E4S: blocked until accelerator-backed runtime evidence or a review channel is available.

Machine-readable evidence remains in `docs/source/_static/registry_submission_receipts.json`.
Target-level closure state remains in `docs/source/_static/external_submission_target_inventory.json`.

Migration source:

- `docs/source/registry_submission_receipts.rst`
