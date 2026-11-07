"""Run a native benchmark policy and write its complete result as JSON."""

import argparse
import json
import os
from pathlib import Path

from innovate import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("fast", "scheduled", "workflow_dispatch"))
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--workers", type=int)
    workers.add_argument("--slurm-allocation", action="store_true",
                         help="Use SLURM_CPUS_PER_TASK as the local worker budget.")
    parser.add_argument("--output", type=Path, required=True)
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
    if args.policy is not None:
        payload["ci_policy"] = args.policy
    response = execute("run_benchmark_suite", **payload)
    response["metadata"]["benchmark_runner"] = {
        "requested_workers": count,
        "worker_budget_source": "SLURM_CPUS_PER_TASK" if args.slurm_allocation else "command_line",
        "ci_policy": args.policy,
    }
    args.output.write_text(json.dumps(response, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = response["result"]
    print(f"Benchmark suite: {result['succeeded']} succeeded, {result['failed']} failed")
    return int(result["failed"] > 0 or not result["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
