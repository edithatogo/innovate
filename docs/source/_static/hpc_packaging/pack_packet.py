"""Generate a consolidated HPC submission packet manifest.

This script is intentionally conservative: it does not attempt to submit to
any registry or execute on a scheduler. It only assembles the package sketches,
scheduler templates, governance templates, and evidence anchors into a single
machine-readable summary that can be copied into a cluster job workspace or
used in release review.
"""

from __future__ import annotations

import json


def build_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifacts": {
            "spack_recipe": "spack/py-innovate.py",
            "easybuild_easyconfig": "easybuild/innovate-0.5.0.eb",
            "scheduler_templates": [
                "scheduler/slurm/spack-smoke.sbatch",
                "scheduler/slurm/easybuild-smoke.sbatch",
                "scheduler/pbs/spack-smoke.pbs",
                "scheduler/pbs/easybuild-smoke.pbs",
            ],
            "governance_templates": [
                "governance/hpsf-evidence.md",
                "governance/e4s-evidence.md",
            ],
            "evidence": [
                "evidence/python-install.log",
                "evidence/python-smoke.log",
                "evidence/r-build.log",
                "evidence/r-check.log",
                "evidence/rust-test.log",
                "evidence/uv-build.log",
                "evidence/julia-installed-smoke.log",
                "evidence/spack-batch.log",
                "evidence/easybuild-batch.log",
                "evidence/spack-pbs.log",
                "evidence/easybuild-pbs.log",
            ],
        },
        "closure_inventory": "docs/source/_static/external_submission_target_inventory.json",
        "targets": [
            {
                "target_id": "spack",
                "registry": "Spack",
                "status": "ready_for_review",
                "submission_mode": "candidate_recipe",
                "packet_artifacts": [
                    "docs/source/_static/hpc_packaging/spack/py-innovate.py",
                    "docs/source/_static/hpc_packaging/scheduler/slurm/spack-smoke.sbatch",
                    "docs/source/_static/hpc_packaging/scheduler/pbs/spack-smoke.pbs",
                    "docs/source/_static/hpc_packaging/pack_packet.py",
                    "docs/source/_static/hpc_packaging/workflow_manifest.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log",
                    "docs/source/_static/hpc_packaging/evidence/python-install.log",
                    "docs/source/_static/hpc_packaging/evidence/python-smoke.log",
                    "docs/source/_static/hpc_packaging/evidence/spack-batch.log",
                    "docs/source/_static/hpc_packaging/evidence/spack-pbs.log",
                ],
                "required_next_step": "Submit the candidate recipe upstream only after maintainer review and scheduler-backed evidence refresh.",
            },
            {
                "target_id": "easybuild",
                "registry": "EasyBuild",
                "status": "ready_for_review",
                "submission_mode": "candidate_easyconfig",
                "packet_artifacts": [
                    "docs/source/_static/hpc_packaging/easybuild/innovate-0.5.0.eb",
                    "docs/source/_static/hpc_packaging/scheduler/slurm/easybuild-smoke.sbatch",
                    "docs/source/_static/hpc_packaging/scheduler/pbs/easybuild-smoke.pbs",
                    "docs/source/_static/hpc_packaging/pack_packet.py",
                    "docs/source/_static/hpc_packaging/workflow_manifest.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log",
                    "docs/source/_static/hpc_packaging/evidence/python-install.log",
                    "docs/source/_static/hpc_packaging/evidence/python-smoke.log",
                    "docs/source/_static/hpc_packaging/evidence/easybuild-batch.log",
                    "docs/source/_static/hpc_packaging/evidence/easybuild-pbs.log",
                ],
                "required_next_step": "Submit the candidate easyconfig upstream only after maintainer review and scheduler-backed evidence refresh.",
            },
            {
                "target_id": "hpsf",
                "registry": "HPSF",
                "status": "ready_for_maintainer",
                "submission_mode": "governance_packet",
                "packet_artifacts": [
                    "docs/source/hpc_registry_contract.rst",
                    "docs/source/hpc_packaging_registry_readiness.rst",
                    "docs/source/_static/hpc_packaging/governance/hpsf-evidence.md",
                    "docs/source/_static/hpc_packaging/evidence/hpsf-review-note.md",
                    "docs/source/_static/hpc_packaging/pack_packet.py",
                    "docs/source/_static/hpc_packaging/workflow_manifest.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log",
                    "docs/source/_static/hpc_packaging/evidence/r-build.log",
                    "docs/source/_static/hpc_packaging/evidence/r-check.log",
                ],
                "required_next_step": "Maintainer must identify two HPSF TAC sponsors, complete the project proposal template, and open the HPSF TAC GitHub proposal issue.",
            },
            {
                "target_id": "e4s",
                "registry": "E4S",
                "status": "ready_for_maintainer",
                "submission_mode": "performance_portability_packet",
                "packet_artifacts": [
                    "docs/source/hpc_registry_contract.rst",
                    "docs/source/hpc_packaging_registry_readiness.rst",
                    "docs/source/_static/hpc_packaging/governance/e4s-evidence.md",
                    "docs/source/_static/hpc_packaging/evidence/e4s-review-note.md",
                    "docs/source/_static/hpc_packaging/pack_packet.py",
                    "docs/source/_static/hpc_packaging/workflow_manifest.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_blockers.json",
                    "docs/source/_static/hpc_packaging/evidence/hpc_submission_environment_probe.log",
                    "docs/source/_static/hpc_packaging/evidence/rust-test.log",
                    "docs/source/_static/hpc_packaging/evidence/julia-installed-smoke.log",
                ],
                "required_next_step": "Maintainer must contact E4S, validate the Spack package through E4S review/CI expectations, and open the inclusion request only after review evidence exists.",
            },
        ],
        "rendered_sources": {
            "packet": "submission_packet.json",
            "readme": "README.md",
        },
    }


def main() -> None:
    manifest = build_manifest()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
