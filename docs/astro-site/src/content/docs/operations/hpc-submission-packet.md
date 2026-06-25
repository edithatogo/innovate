---
title: HPC Submission Packet
description: Targeted evidence packets for remaining HPC registry work.
---

# HPC Submission Packet

This page is the execution packet for HPC registration paths.
The packet is not a submission claim; it is the executable list of artifacts
needed to move each target from readiness into upstream review.

Current status matrix:

- Spack: ready_for_review, pending maintainer-managed upstream submission.
- EasyBuild: ready_for_review, pending maintainer-managed upstream submission.
- HPSF: ready_for_maintainer; public proposal path identified, pending TAC sponsors and maintainer-opened proposal issue.
- E4S: ready_for_maintainer; public contact/contribution path identified, pending E4S review/CI validation and maintainer inclusion request.

For each target, the repository stores command paths, candidate package artifacts, and evidence links under `docs/source/_static/hpc_packaging/`.
Target-level closure state is tracked in `docs/source/_static/external_submission_target_inventory.json`.

Machine-readable packet:

- `docs/source/_static/hpc_packaging/submission_packet.json`
- `docs/source/_static/hpc_packaging/workflow_manifest.json`
- `docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json`
- `docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log`

Execution templates and evidence anchors:

- `scheduler/slurm/spack-smoke.sbatch`
- `scheduler/slurm/easybuild-smoke.sbatch`
- `scheduler/pbs/spack-smoke.pbs`
- `scheduler/pbs/easybuild-smoke.pbs`
- `governance/hpsf-evidence.md`
- `governance/e4s-evidence.md`
- `pack_packet.py`
- `spack-batch.log`
- `easybuild-batch.log`
- `spack-pbs.log`
- `easybuild-pbs.log`
- `hpsf-review-note.md`
- `e4s-review-note.md`

Each target records maintainer owner, external action URL, requirement sources,
receipt rule, and revisit condition.
