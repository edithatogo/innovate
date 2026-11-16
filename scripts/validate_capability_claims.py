"""Validate conductor/capabilities.json evidence claims (R10)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate_capabilities(
    data: dict[str, Any],
    *,
    expected_revision: str | None = None,
    root: Path = Path("."),
) -> list[str]:
    """Validate capability records; returns a list of error strings (empty if valid)."""
    errors: list[str] = []
    if data.get("schema_version") != "1.0":
        errors.append("unsupported capabilities schema_version")
    rev = data.get("revision")
    if not isinstance(rev, str) or not rev.strip():
        errors.append("missing repository revision")
    elif expected_revision is not None and rev != expected_revision:
        errors.append(f"stale revision: {rev} != {expected_revision}")

    caps = data.get("capabilities")
    if not isinstance(caps, list) or not caps:
        errors.append("capabilities must be a non-empty list")
        return errors

    for idx, cap in enumerate(caps):
        label = f"capabilities[{idx}]"
        for field in ("model", "operation", "payload", "backend"):
            if not isinstance(cap.get(field), str) or not cap[field].strip():
                errors.append(f"{label} missing {field}")
        ev = cap.get("evidence")
        if not isinstance(ev, dict):
            errors.append(f"{label} missing evidence object")
            continue
        if cap.get("backend") == "rust_native":
            receipt = ev.get("native_receipt")
            if not isinstance(receipt, str) or not receipt.strip():
                errors.append(f"{label} claims rust_native without native_receipt")
            elif not (root / receipt).exists():
                errors.append(f"{label} native_receipt does not exist: {receipt}")
        for receipt_key in ("scientific_receipt", "empirical_receipt", "release_receipt"):
            val = ev.get(receipt_key)
            if val is not None and (not isinstance(val, str) or not (root / val).exists()):
                errors.append(f"{label} {receipt_key} does not exist: {val}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate capability claims matrix.")
    parser.add_argument("path", type=Path, default=Path("conductor/capabilities.json"), nargs="?")
    args = parser.parse_args()
    data = json.loads(args.path.read_text(encoding="utf-8"))
    errors = validate_capabilities(data)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print(f"Capabilities matrix valid: {len(data['capabilities'])} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
