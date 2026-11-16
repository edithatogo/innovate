"""Check the registry identities used to audit vendored Rust source.

Path dependencies are skipped by RustSec's registry advisory matcher.  The
separate lock is an audit inventory, not a Cargo resolution or a release lock.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[1] / "rust/vendor"
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"


def audit_lock(vendor: Path) -> str:
    inventory = json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("packages"), list):
        raise ValueError("invalid upstream inventory")
    records = []
    directories = set()
    for package in inventory["packages"]:
        name = package["name"]
        version = package["version"]
        checksum = package["crate_sha256"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not re.fullmatch(
            r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", version
        ):
            raise ValueError("invalid upstream package identity")
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"invalid upstream checksum for {name}")
        directory = f"{name}-{version}"
        if directory in directories:
            raise ValueError(f"duplicate upstream package {directory}")
        directories.add(directory)
        with (vendor / directory / "Cargo.toml").open("rb") as stream:
            manifest = tomllib.load(stream)["package"]
        if (manifest["name"], manifest["version"]) != (name, version):
            raise ValueError(f"manifest identity differs for {directory}")
        records.append((name, version, checksum))
    actual = {path.name for path in vendor.iterdir() if path.is_dir()}
    if not records or actual != directories:
        raise ValueError("vendored directories and upstream inventory differ")
    lines = ["# Registry identities of vendored sources, for advisory audits only.", "version = 4", ""]
    for name, version, checksum in sorted(records):
        lines.extend(
            [
                "[[package]]",
                f"name = {json.dumps(name)}",
                f"version = {json.dumps(version)}",
                f"source = {json.dumps(REGISTRY)}",
                f"checksum = {json.dumps(checksum)}",
                "",
            ]
        )
    return "\n".join(lines)


def check_inventory(vendor: Path) -> None:
    expected = audit_lock(vendor)
    actual = (vendor / "Cargo.lock").read_text(encoding="utf-8")
    if actual != expected:
        raise ValueError("vendored audit lock differs from upstream inventory")


def main() -> int:
    try:
        check_inventory(VENDOR)
    except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        print(f"Vendored advisory inventory: FAIL: {error}", file=sys.stderr)
        return 1
    print("Vendored advisory inventory: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
