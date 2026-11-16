"""Fail closed on capability claims without qualified source-bound receipts.

The existing matrix holds source-file paths, not execution or acceptance
receipts. A future integrator-owned receipt contract must bind evidence bytes,
the exact source and capability before this checker can qualify those claims.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import typing


def _checkout_revision(root: pathlib.Path) -> str | None:
    """Return an exact checkout HEAD only when root is its actual Git root."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if pathlib.Path(top).resolve() != root.resolve():
            return None
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return None
    return head if re.fullmatch(r"[0-9a-f]{40}", head) else None


def _has_source_commit(root: pathlib.Path, revision: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{revision}^{{commit}}"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return False
    return result.returncode == 0


def _receipt_issue(root: pathlib.Path, value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return f"{label} is missing"
    try:
        path = pathlib.Path(value)
        if path.is_absolute() or ".." in path.parts:
            return f"{label} points outside the checkout"
        resolved_root = root.resolve()
        resolved_path = (resolved_root / path).resolve()
        if not resolved_path.is_relative_to(resolved_root):
            return f"{label} points outside the checkout"
        if not resolved_path.is_file():
            return f"{label} does not exist as a file in the checkout"
    except OSError, RuntimeError, ValueError:
        return f"{label} has an invalid path"
    # A path, extension or self-reported pass does not bind bytes to the
    # capability and source. No receipt schema has been qualified here.
    return f"{label} is not a qualified source-bound receipt"


def validate_capabilities(
    data: dict[str, typing.Any],
    *,
    expected_revision: str | None = None,
    root: pathlib.Path = pathlib.Path("."),
) -> list[str]:
    """Return qualification errors; existing unbound claims remain unavailable."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["capability matrix must be an object"]
    if data.get("schema_version") != "1.0":
        errors.append("unsupported capabilities schema_version")
    checkout = _checkout_revision(root)
    if checkout is None:
        errors.append("checkout revision cannot be verified")
    revision = data.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        errors.append("matrix revision must be a full source commit")
    elif checkout is not None and not _has_source_commit(root, revision):
        errors.append("matrix revision is not a locally available source commit")
    if expected_revision is None:
        errors.append("expected source revision is required for qualification")
    elif not isinstance(expected_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        errors.append("expected source revision must be a full source commit")
    elif revision != expected_revision:
        errors.append("matrix revision does not match the expected source revision")

    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        errors.append("capabilities must be a non-empty list")
        return errors
    for idx, cap in enumerate(capabilities):
        label = f"capabilities[{idx}]"
        if not isinstance(cap, dict):
            errors.append(f"{label} must be an object")
            continue
        for field in ("model", "operation", "payload", "backend"):
            if not isinstance(cap.get(field), str) or not cap[field].strip():
                errors.append(f"{label} missing {field}")
        evidence = cap.get("evidence")
        if not isinstance(evidence, dict):
            errors.append(f"{label} missing evidence object")
            continue
        if cap.get("backend") == "rust_native":
            errors.append(_receipt_issue(root, evidence.get("native_receipt"), f"{label} native_receipt"))
        else:
            errors.append(f"{label} backend is unsupported by the qualified checker")
        for key in ("scientific_receipt", "empirical_receipt", "release_receipt"):
            if evidence.get(key) is not None:
                errors.append(_receipt_issue(root, evidence[key], f"{label} {key}"))
    return errors


def main() -> int:
    """Report unqualified capability claims from the selected checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path, default=pathlib.Path("conductor/capabilities.json"), nargs="?")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--source-revision", help="Full source commit expected by the qualification")
    args = parser.parse_args()
    try:
        data = json.loads(args.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Cannot read capability matrix: {exc}\n")
    errors = validate_capabilities(data, expected_revision=args.source_revision, root=args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Capabilities matrix valid: {len(data['capabilities'])} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
