"""Run a native benchmark policy and write its complete result as JSON."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

from innovate import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("fast", "scheduled", "workflow_dispatch"))
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--workers", type=int)
    workers.add_argument("--slurm-allocation", action="store_true",
                         help="Use SLURM_CPUS_PER_TASK as the local worker budget.")
    parser.add_argument("--mpi-shard", action="store_true", help="Select cases for this Open MPI rank.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Receipt path. With --mpi-shard, {rank} in the filename expands to the rank.")
    args = parser.parse_args()

    count = args.workers if args.workers is not None else 1
    if args.slurm_allocation:
        allocated = os.environ.get("SLURM_CPUS_PER_TASK", "")
        try:
            count = int(allocated)
        except ValueError:
            parser.error("--slurm-allocation requires a positive integer SLURM_CPUS_PER_TASK")
        if count < 1:
            parser.error("--slurm-allocation requires a positive integer SLURM_CPUS_PER_TASK")
        modes = execute("discover_execution_modes")["result"]
        count = min(count, modes["local_cpu"]["max_workers"])
    payload = {"workers": count}
    shard = None
    if args.mpi_shard:
        try:
            rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        except (KeyError, ValueError):
            parser.error("--mpi-shard requires Open MPI rank and size")
        cases = execute("discover_benchmark_cases")["result"]["cases"]
        ids = [case["case_id"] for case in cases if args.policy is None or case["ci_policy"] == args.policy]
        if not (0 <= rank < size <= len(ids)):
            parser.error("--mpi-shard requires a valid rank and at least one case per rank")
        payload["case_ids"] = ids[rank::size]
        shard = {"rank": rank, "size": size, "case_ids": payload["case_ids"]}
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
    response["metadata"]["benchmark_runner"] = {
        "requested_workers": count,
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
    try:
        with output.open("x" if shard is not None else "w", encoding="utf-8") as stream:
            stream.write(receipt)
    except FileExistsError:
        parser.error(f"MPI receipt already exists: {output}; use a fresh path for each rank")
    result = response["result"]
    print(f"Benchmark suite: {result['succeeded']} succeeded, {result['failed']} failed")
    return int(result["failed"] > 0 or not result["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
