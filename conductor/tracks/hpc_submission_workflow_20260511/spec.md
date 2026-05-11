# Specification: HPC Submission Workflow Arrangement and Registry Handoff

## Overview

Create a concrete, auditable workflow for the remaining HPC submission work
covering Spack, EasyBuild, HPSF, and E4S. The repository already contains the
candidate package sketches, evidence logs, scheduler templates, governance
templates, and packet generator needed to arrange the work; this track turns
those artifacts into an execution-ready handoff and records the actual external
submission outcomes or blocker states.

## Background

The repository has HPC readiness docs and a submission packet, but the blocked
targets still require a real cluster run or governance submission to complete.
This track focuses on arranging each target with the correct artifacts, running
the submission workflow where possible, and preserving auditable evidence in
the repo.

## Functional Requirements

1. Maintain a single machine-readable HPC packet for the blocked targets.
2. Preserve scheduler templates for Slurm and PBS runs of the Spack and
   EasyBuild candidates.
3. Preserve governance templates for the HPSF and E4S evidence bundles.
4. Record a per-target checklist describing the command, artifacts, and next
   step needed for each HPC registry path.
5. Execute or prepare the external handoff for each target:
   - Spack recipe submission or review
   - EasyBuild easyconfig submission or review
   - HPSF governance packet handoff
   - E4S performance-portability packet handoff
6. Capture scheduler metadata, batch logs, registry URLs, review IDs, or
   blocker notes for each target.
7. Update the readiness docs and machine-readable packet so they distinguish
   prepared, submitted, blocked, and deferred states without overstating
   success.

## Non-Functional Requirements

1. The workflow must remain reproducible from repo artifacts.
2. The track must not claim submission success without a durable external
   reference, receipt, or batch log.
3. The packet and docs must remain synchronized.
4. The track must not introduce new package ecosystems or change the project
   source layout.

## Acceptance Criteria

1. Each HPC target has a documented execution path and current status.
2. Spack and EasyBuild have scheduler-backed run evidence or a clearly stated
   blocker note.
3. HPSF and E4S have governance or performance-portability packet evidence or
   a clearly stated blocker note.
4. The packet generator, scheduler templates, governance templates, and docs
   all point to the same state.
5. Tests assert that the HPC packet and templates remain present and
   non-claiming until external evidence is captured.
6. The track can be archived once all target states are fully reconciled.

## Out of Scope

1. General product feature work unrelated to HPC packaging.
2. New bindings or runtime changes.
3. Non-HPC registry submission work already tracked elsewhere.
4. Reorganizing the source tree or documentation architecture.
