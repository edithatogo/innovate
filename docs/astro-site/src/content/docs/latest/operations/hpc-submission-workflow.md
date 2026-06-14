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

Status boundary:

* Spack and EasyBuild are `ready_for_review`, not submitted.
* HPSF and E4S remain blocked on governance and accelerator-review evidence.
* Target-level closure state is tracked in `docs/source/_static/external_submission_target_inventory.json`.

Migration source:

* `docs/source/hpc_submission_workflow.rst`
