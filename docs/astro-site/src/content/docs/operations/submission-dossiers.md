---
title: Submission Readiness Dossiers
description: Evidence dossiers for submission and governance targets.
---

# Submission Readiness Dossiers

This page is the consolidated entry point for reviewer-facing dossier evidence.

Primary sections include pyOpenSci, rOpenSci, JOSS, NumFOCUS, community targets, and the full registry matrix evidence.

The page is intentionally evidence-driven: each target points to concrete files and release artifacts rather than aspirational roadmap text.

Current target-level closure state is recorded in
`docs/source/_static/external_submission_target_inventory.json`; readiness does
not imply external submission or acceptance.
The receipt and owner-backed deferral ledger is
`docs/source/_static/external_acceptance_deferrals.json`.
The CRAN/scientific handoff packet is
`docs/source/_static/scientific_submission_packet.json`.

## CRAN and scientific submission packet

The CRAN and scientific submission packet is the maintainer-ready handoff for
CRAN, pyOpenSci, rOpenSci, JOSS, NumFOCUS, PyPA, Apache Arrow, .NET Foundation,
Julia community, R community, and scikit-learn-contrib decisions. It records
the maintainer owner, external action URL, local artifacts, receipt rule, and
revisit condition for each target.

## Reviewer evidence

- pyOpenSci: README, API docs, tutorials, issue templates, CI, contribution, and security evidence.
- rOpenSci: R package metadata, README, `cran-comments.md`, manuals, vignettes, tests, and publication gates.
- JOSS: `paper.qmd`, manuscript sources, tutorials, benchmark and diagnostics coverage, citation and governance metadata.
- Apache Arrow: ADRs, Arrow interchange tutorial, ABI strategy, and Arrow contract tests.
- .NET Foundation: C# binding docs, package model, publication gates, support matrix, and governance dossier.
- NumFOCUS: governance, support, funding, sustainability, roadmap, and stewardship evidence.
- R community and Julia community: language binding docs, examples, tests, registry notes, and maintenance owners.
- scikit-learn-contrib: deliberately not applicable unless project scope changes.

The packet does not claim external submission or acceptance.
