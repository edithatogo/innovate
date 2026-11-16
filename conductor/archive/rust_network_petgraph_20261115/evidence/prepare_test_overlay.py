"""Prepare an external test package without editing shared repository manifests.

Usage: python prepare_test_overlay.py /absolute/new/temporary/directory
Next, reconcile only the external copied lock (do not pass --locked yet):
cargo metadata --offline --manifest-path /absolute/new/temporary/directory/Cargo.toml --format-version 1
The recorded --locked test commands then apply to that external package.
The generated package compiles the actual core source and selected real tests.
Its sole added dependency is the pending integrator request. This is conditional
source verification, not a repository --locked or release qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib


def toml(value: object) -> str:
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {toml(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(toml(v) for v in value) + "]"
    return json.dumps(value)


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    destination = Path(sys.argv[1]).resolve()
    if destination.is_relative_to(root):
        raise SystemExit("The test package must be outside the repository")
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    core = root / "rust/crates/innovate-core"
    package = tomllib.loads((core / "Cargo.toml").read_text())
    dependencies = {
        name: workspace["workspace"]["dependencies"][name]
        if isinstance(spec, dict) and spec.get("workspace")
        else spec
        for name, spec in package["dependencies"].items()
    }
    if "petgraph" in dependencies:
        raise SystemExit("Direct petgraph is already declared; use repository checks")
    # --baseline is for dependency-metadata comparison only once source imports
    # petgraph; it deliberately does not satisfy that pending direct dependency.
    if "--baseline" not in sys.argv[2:]:
        dependencies["petgraph"] = {
            "version": "=0.8.3",
            "default-features": False,
        }
    text = ['[workspace]', 'resolver = "3"', "", "[package]", 'name = "innovate-core"']
    text.extend(
        f"{key} = {toml(workspace['workspace']['package'][key])}"
        for key in ("version", "edition", "rust-version", "license")
    )
    text.extend(["", "[lib]", f"path = {toml(str(core / 'src/lib.rs'))}"])
    text.extend(["", "[dependencies]"])
    text.extend(f"{name} = {toml(spec)}" for name, spec in dependencies.items())
    for target in ("network", "network_petgraph", "network_diffusion", "kairos"):
        test_path = core / "tests" / (target + ".rs")
        text.extend(["", "[[test]]", f"name = {toml(target)}", f"path = {toml(str(test_path))}"])
    text.extend(["", "[profile.dev]", "debug = 0", "", "[profile.test]", "debug = 0"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "Cargo.toml").write_text("\n".join(text) + "\n")
    shutil.copyfile(root / "Cargo.lock", destination / "Cargo.lock")
    sources = sorted(
        p for part in ("src", "tests") for p in (core / part).rglob("*") if p.is_file()
    )
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    (destination / "source-hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    print(destination / "Cargo.toml")


if __name__ == "__main__":
    main()
