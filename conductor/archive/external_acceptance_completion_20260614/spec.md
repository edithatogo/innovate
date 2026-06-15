# External Acceptance Completion

## Overview

The current repository distinguishes readiness from acceptance. This track pushes every feasible external registry, scientific community, and HPC target to actual submission or accepted status where maintainer authority and credentials allow. Targets that cannot be completed must be converted into durable, owner-backed deferrals with exact next actions and evidence.

## Functional Requirements

1. Refresh live evidence for all package-manager registries and pending submission targets.
2. Complete or prepare maintainer-approved CRAN submission and record the receipt.
3. Refresh Spack and EasyBuild scheduler/package evidence and prepare upstream submission PRs.
4. Prepare HPSF and E4S maintainer action packets with sponsor/contact requirements, proposal text, and evidence links.
5. Prepare scientific/community submission packets for pyOpenSci, rOpenSci, JOSS, NumFOCUS, Arrow, .NET Foundation, Julia community, and R community.
6. Update registry inventories, receipts, and docs so every target is accepted, submitted, ready-for-maintainer, deferred, or not applicable with evidence.

## Non-Functional Requirements

1. Do not falsely claim acceptance where only readiness exists.
2. Do not perform external final-submit actions that require maintainer approval unless approval and credentials are present.
3. Every manual gate must have a prepared artifact, owner, exact URL or contact path, and revisit condition.
4. External evidence should be refreshed close to submission time.

## Acceptance Criteria

1. All feasible submissions have receipts or prepared external action packets.
2. CRAN, Spack, EasyBuild, HPSF, and E4S no longer have generic blocker language.
3. Scientific/community paths have final owner-backed states.
4. Machine-readable inventories and Starlight docs agree.
5. GitHub Actions passes after evidence refresh and docs updates.

## Required Operational Cadence

Every task requires a task implementation commit, a separate plan-status commit, phase review with `conductor-review`, push plus GitHub Actions monitoring, final track review, final push, and passing GitHub Actions before archive.

## Out of Scope

1. Clicking final external submission buttons without maintainer approval.
2. Paying fees or accepting legal terms on behalf of maintainers.
3. Claiming acceptance before an external URL, receipt, merged PR, or official response exists.
