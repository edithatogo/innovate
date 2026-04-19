"""Kernel bridge entrypoint for Innovate Rust bindings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from innovate.kernel import (
    KernelRequest,
    discover_models,
    diagnose_model,
    fit_model,
    predict_model,
    simulate_model,
    summarize_model,
)

OPERATIONS = {
    "discover_models": discover_models,
    "fit_model": fit_model,
    "predict_model": predict_model,
    "simulate_model": simulate_model,
    "summarize_model": summarize_model,
    "diagnose_model": diagnose_model,
}


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: kernel_bridge.py <request.json> <response.json>")

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = KernelRequest.from_dict(json.loads(request_path.read_text()))
    operation = OPERATIONS[request.operation]
    response = operation() if request.operation == "discover_models" else operation(request)
    if request.operation == "discover_models":
        payload = {
            "schema_version": request.schema_version,
            "operation": request.operation,
            "model_key": None,
            "result": response.to_dict(),
            "error": None,
            "metadata": {},
        }
    else:
        payload = response.to_dict()
    response_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
