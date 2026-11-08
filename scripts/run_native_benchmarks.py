"""Run a native benchmark policy and write its complete result as JSON."""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from innovate import _native, execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("fast", "scheduled", "workflow_dispatch"))
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--workers", type=int)
    workers.add_argument("--slurm-allocation", action="store_true",
                         help="Use SLURM_CPUS_PER_TASK as the local worker budget.")
    launchers = parser.add_mutually_exclusive_group()
    launchers.add_argument("--mpi-shard", action="store_true", help="Select cases for this Open MPI rank.")
    launchers.add_argument("--slurm-shard", action="store_true", help="Select cases for this Slurm task.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Receipt path. For sharded runs, {rank} in the filename expands to the rank.")
    args = parser.parse_args()

    count = args.workers if args.workers is not None else 1
    allocated_cpus = None
    if args.slurm_allocation:
        allocated = os.environ.get("SLURM_CPUS_PER_TASK", "")
        try:
            count = int(allocated)
        except ValueError:
            parser.error("--slurm-allocation requires a positive integer SLURM_CPUS_PER_TASK")
        if count < 1:
            parser.error("--slurm-allocation requires a positive integer SLURM_CPUS_PER_TASK")
        modes = execute("discover_execution_modes")["result"]
        allocated_cpus = count
        count = min(count, modes["local_cpu"]["max_workers"])
    payload = {"workers": count}
    shard = None
    if args.mpi_shard or args.slurm_shard:
        rank_key, size_key = (
            ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE")
            if args.mpi_shard else ("SLURM_PROCID", "SLURM_STEP_NUM_TASKS")
        )
        try:
            rank = int(os.environ[rank_key])
            size = int(os.environ[size_key])
        except (KeyError, ValueError):
            parser.error(f"sharding requires integer {rank_key} and {size_key}")
        cases = execute("discover_benchmark_cases")["result"]["cases"]
        ids = [case["case_id"] for case in cases if args.policy is None or case["ci_policy"] == args.policy]
        if not (0 <= rank < size <= len(ids)):
            parser.error("sharding requires a valid rank and at least one case per rank")
        payload["case_ids"] = ids[rank::size]
        shard = {"rank": rank, "size": size, "case_ids": payload["case_ids"],
                 "launcher": "open_mpi" if args.mpi_shard else "slurm"}
        if args.slurm_shard:
            job_id = os.environ.get("SLURM_JOB_ID", "").strip()
            step_id = os.environ.get("SLURM_STEP_ID", "").strip()
            if not job_id or not step_id:
                parser.error("--slurm-shard requires SLURM_JOB_ID and SLURM_STEP_ID")
            shard.update(job_id=job_id, step_id=step_id)
    elif args.policy is not None:
        payload["ci_policy"] = args.policy
    started = perf_counter()
    response = execute("run_benchmark_suite", **payload)
    elapsed = perf_counter() - started
    system = platform.system()
    peak_rss_bytes = None
    if system in {"Darwin", "Linux"}:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = peak_rss if system == "Darwin" else peak_rss * 1024
    with Path(_native.__file__).open("rb") as extension:
        native_extension_sha256 = hashlib.file_digest(extension, "sha256").hexdigest()
    response["metadata"]["benchmark_runner"] = {
        "native_extension_sha256": native_extension_sha256,
        "requested_workers": count,
        "slurm_allocated_cpus_per_task": allocated_cpus,
        "worker_budget_source": "SLURM_CPUS_PER_TASK" if args.slurm_allocation else "command_line",
        "ci_policy": args.policy,
        "kernel_elapsed_seconds": elapsed,
        "process_peak_rss_bytes": peak_rss_bytes,
        "python_version": platform.python_version(),
        "system": system,
        "architecture": platform.machine(),
    }
    output = args.output
    if shard is not None:
        response["metadata"]["benchmark_runner"]["mpi_shard"] = shard
        output = output.with_name(output.name.replace("{rank}", str(shard["rank"])))
    receipt = json.dumps(response, indent=2, allow_nan=False) + "\n"
    if shard is None:
        output.write_text(receipt, encoding="utf-8")
    else:
        try:
            # A same-filesystem hard link publishes the complete file without replacing an existing rank.
            with TemporaryDirectory(prefix=".innovate-receipt-", dir=output.parent) as staging:
                staged = Path(staging) / "receipt.json"
                staged.write_text(receipt, encoding="utf-8")
                os.link(staged, output)
        except FileExistsError:
            parser.error(f"Shard receipt already exists: {output}; use a fresh path for each rank")
    result = response["result"]
    print(f"Benchmark suite: {result['succeeded']} succeeded, {result['failed']} failed")
    return int(result["failed"] > 0 or not result["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
