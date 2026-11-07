"""Verify MPI benchmark receipts against a serial receipt from the same build."""

import argparse
import json
from pathlib import Path


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
    ranks = set()
    for shard in shards:
        actual = shard["metadata"]
        config = actual["benchmark_runner"]
        if (actual["kernel_version"], config["native_extension_sha256"], config["ci_policy"]) != (
            metadata["kernel_version"],
            runner["native_extension_sha256"],
            runner["ci_policy"],
        ):
            raise ValueError("shards must match the serial kernel version, extension digest and policy")
        partition = config["mpi_shard"]
        rank = partition["rank"]
        size = partition["size"]
        if (
            type(size) is not int
            or type(rank) is not int
            or size != len(shards)
            or not 0 <= rank < size
            or rank in ranks
        ):
            raise ValueError("missing, duplicate or invalid MPI rank")
        ranks.add(rank)
        rows = checked_runs(shard)
        if rows != expected[rank :: len(shards)] or partition["case_ids"] != [row["case_id"] for row in rows]:
            raise ValueError("MPI rank case IDs or results differ from the serial partition")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        check(json.loads(args.serial.read_text()), [json.loads(path.read_text()) for path in args.shards])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Verified {len(args.shards)} MPI rank receipts against serial results")


if __name__ == "__main__":
    main()
