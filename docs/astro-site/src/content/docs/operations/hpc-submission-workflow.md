---
title: HPC Submission Workflow
description: Per-target execution checklist for HPC registry submission.
---

# HPC Submission Workflow

The HPC workflow is now split by target:

- Spack recipe execution and log capture.
- EasyBuild easyconfig execution and sanity logs.
- HPSF governance packet completion.
- E4S portability packet preparation.

Run the command bundles documented by the manifest and record outputs in
`docs/source/_static/hpc_packaging/evidence/`.

Status boundary:

- Spack and EasyBuild are `ready_for_review`, not submitted.
- HPSF and E4S are `ready_for_maintainer`, not submitted; external proposal,
  sponsor, contact, and review/CI validation steps remain maintainer-managed.
- Target-level closure state is tracked in `docs/source/_static/external_submission_target_inventory.json`.

Migration source:

- `docs/source/hpc_submission_workflow.rst`
