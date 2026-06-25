---
title: Community Submission Readiness
description: Scientific reviewer-facing readiness dossier for community targets.
slug: latest/operations/community-readiness
---

# Community Submission Readiness

This page maps evidence readiness for scientific targets and ties each target to repository artifacts.

Covered communities include pyOpenSci, rOpenSci, JOSS, NumFOCUS, scikit-learn-contrib, Apache Arrow, Julia community, R community, .NET Foundation, and Julia General.

No submission claims readiness without evidence; every target records explicit status, evidence links, blockers, and sequencing.
Readiness is not submission. Targets marked ready remain `ready_for_review`
unless a durable external review URL, receipt, or acceptance record exists.

Machine-readable closure state is tracked in
`docs/source/_static/external_submission_target_inventory.json`, which keeps
readiness separate from actual external submission or acceptance.
The reviewer matrix lives at
`docs/source/_static/community_submission_readiness_matrix.json`.

## Submission sequencing

1. pyOpenSci and Apache Arrow community conversations.
2. R community and rOpenSci after R package evidence is current.
3. JOSS after paper metadata, citation, statement of need, and reproducible examples are finalized.
4. Julia community after installed-package bridge expectations and registration notes are stable.
5. .NET Foundation and NumFOCUS after governance and sustainability evidence is complete.
6. scikit-learn-contrib only if the project later decides estimator-ecosystem scope is appropriate.

Reviewer evidence includes docs, tests, examples, citation, governance, and
maintenance evidence. Targets with missing or dependent evidence are marked
`near_ready` or `not_applicable` rather than ready.
