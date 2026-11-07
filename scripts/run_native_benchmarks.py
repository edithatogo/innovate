"""Run a native benchmark policy and write its complete result as JSON."""

import argparse
import json
from pathlib import Path

from innovate import execute


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("fast", "scheduled", "workflow_dispatch"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {"workers": args.workers}
    if args.policy is not None:
        payload["ci_policy"] = args.policy
    response = execute("run_benchmark_suite", **payload)
    args.output.write_text(json.dumps(response, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = response["result"]
    print(f"Benchmark suite: {result['succeeded']} succeeded, {result['failed']} failed")
    return int(result["failed"] > 0 or not result["runs"])


if __name__ == "__main__":
    raise SystemExit(main())
