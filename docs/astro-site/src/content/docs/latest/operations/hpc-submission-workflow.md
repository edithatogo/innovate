---
title: HPC Submission Workflow
description: Per-target execution checklist for HPC registry submission.
slug: latest/operations/hpc-submission-workflow
---

# HPC Submission Workflow

The HPC workflow is now split by target:

* Spack recipe execution and log capture.
* EasyBuild easyconfig execution and sanity logs.
* HPSF governance packet completion.
* E4S portability packet preparation.

Run the command bundles documented by the manifest and record outputs in
`docs/source/_static/hpc_packaging/evidence/`.
The machine-readable manifest lives at
`docs/source/_static/hpc_packaging/workflow_manifest.json`; the
compatibility-named blocker bundle is
`docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json`, and
the local probe is
`docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log`.

Status boundary:

* Spack and EasyBuild are `ready_for_review`, not submitted.
* HPSF and E4S are `ready_for_maintainer`, not submitted; external proposal,
  sponsor, contact, and review/CI validation steps remain maintainer-managed.
* Target-level closure state is tracked in `docs/source/_static/external_submission_target_inventory.json`.

## Commands

* Spack: `sbatch docs/source/_static/hpc_packaging/scheduler/slurm/spack-smoke.sbatch`
* EasyBuild: `sbatch docs/source/_static/hpc_packaging/scheduler/slurm/easybuild-smoke.sbatch`
* HPSF: edit `docs/source/_static/hpc_packaging/governance/hpsf-evidence.md` and preserve `evidence/hpsf-review-note.md`.
* E4S: edit `docs/source/_static/hpc_packaging/governance/e4s-evidence.md` and preserve `evidence/e4s-review-note.md`.

The workflow documents Handoff status and maintainer handoff states. The
blocker bundle is now a compatibility-named closure artifact.
