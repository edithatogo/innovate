"""Thin JSON bridge for the Go bindings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from innovate import kernel


def _load_request(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("Usage: kernel_bridge.py <request.json> <response.json>")

    request_path = Path(argv[1])
    response_path = Path(argv[2])

    try:
        request_data = _load_request(request_path)
        request = kernel.KernelRequest.from_dict(request_data)
        dispatch = {
            "discover_models": kernel.discover_models,
            "fit_model": kernel.fit_model,
            "predict_model": kernel.predict_model,
            "simulate_model": kernel.simulate_model,
            "summarize_model": kernel.summarize_model,
            "diagnose_model": kernel.diagnose_model,
        }
        response = (
            dispatch[request.operation](request) if request.operation != "discover_models" else kernel.discover_models()
        )
    except Exception as exc:  # pragma: no cover - defensive bridge fallback
        response = kernel.KernelResponse(
            operation=str(request_data.get("operation", "discover_models"))
            if "request_data" in locals()
            else "discover_models",
            error=kernel.KernelError(
                code=kernel.KernelErrorCode.INVALID_REQUEST.value,
                message=str(exc),
            ),
        )

    _write_response(response_path, response.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
