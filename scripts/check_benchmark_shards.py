"""Verify sharded benchmark receipts against a serial receipt from the same build."""

import argparse
import hashlib
import json
from pathlib import Path

from innovate import _native, execute


def check(serial, shards):
    def checked_runs(receipt):
        result = receipt["result"]
        rows = result["runs"]
        if receipt["error"] is not None or result["failed"] or not rows or result["succeeded"] != len(rows):
            raise ValueError("benchmark receipt did not complete successfully")
        if any(row["error"] is not None for row in rows):
            raise ValueError("benchmark receipt contains a failed case")
        return rows

    expected = checked_runs(serial)
    metadata = serial["metadata"]
    runner = metadata["benchmark_runner"]
    if "mpi_shard" in runner:
        raise ValueError("the reference receipt must come from a serial run")
    if len({row["case_id"] for row in expected}) != len(expected):
        raise ValueError("the serial receipt contains duplicate case IDs")
    with Path(_native.__file__).open("rb") as extension:
        digest = hashlib.file_digest(extension, "sha256").hexdigest()
    if digest != runner["native_extension_sha256"]:
        raise ValueError("the installed native extension must match the serial receipt")
    catalog = execute("discover_benchmark_cases")
    policy = runner["ci_policy"]
    case_ids = [case["case_id"] for case in catalog["result"]["cases"] if policy is None or case["ci_policy"] == policy]
    if [row["case_id"] for row in expected] != case_ids:
        raise ValueError("the serial receipt does not cover the complete benchmark policy")
    max_workers = execute("discover_execution_modes")["result"]["local_cpu"]["max_workers"]

    def check_worker_budget(config):
        if "slurm_allocated_cpus_per_task" not in config:
            return  # Older receipts did not retain the uncapped allocation.
        allocated = config["slurm_allocated_cpus_per_task"]
        if config["worker_budget_source"] == "SLURM_CPUS_PER_TASK":
            if (type(allocated) is not int or allocated < 1
                    or config["requested_workers"] != min(allocated, max_workers)):
                raise ValueError("Slurm worker request does not match the capped allocation")
        elif allocated is not None:
            raise ValueError("only Slurm worker budgets may record allocated CPUs")

    check_worker_budget(runner)
    ranks = set()
    launchers = set()
    slurm_steps = set()
    for shard in shards:
        actual = shard["metadata"]
        config = actual["benchmark_runner"]
        if (actual["kernel_version"], config["native_extension_sha256"], config["ci_policy"]) != (
            metadata["kernel_version"],
            runner["native_extension_sha256"],
            runner["ci_policy"],
        ):
            raise ValueError("shards must match the serial kernel version, extension digest and policy")
        check_worker_budget(config)
        partition = config["mpi_shard"]
        # Receipts before Slurm support were always produced by Open MPI.
        launcher = partition.get("launcher", "open_mpi")
        if launcher not in ("open_mpi", "slurm"):
            raise ValueError("unknown benchmark shard launcher")
        launchers.add(launcher)
        if len(launchers) > 1:
            raise ValueError("benchmark shards must use the same launcher")
        if launcher == "slurm":
            step = (partition.get("job_id"), partition.get("step_id"))
            if any(not isinstance(value, str) or not value.strip() for value in step):
                raise ValueError("Slurm shards require job and step IDs")
            slurm_steps.add(step)
            if len(slurm_steps) > 1:
                raise ValueError("Slurm shards must come from the same job and step")
        rank = partition["rank"]
        size = partition["size"]
        if (
            type(size) is not int
            or type(rank) is not int
            or size != len(shards)
            or not 0 <= rank < size
            or rank in ranks
        ):
            raise ValueError("missing, duplicate or invalid shard rank")
        ranks.add(rank)
        rows = checked_runs(shard)
        if rows != expected[rank :: len(shards)] or partition["case_ids"] != [row["case_id"] for row in rows]:
            raise ValueError("shard rank case IDs or results differ from the serial partition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        check(json.loads(args.serial.read_text()), [json.loads(path.read_text()) for path in args.shards])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified {len(args.shards)} shard rank receipts against serial results")


if __name__ == "__main__":
    main()
