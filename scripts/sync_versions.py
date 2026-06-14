#!/usr/bin/env python3
"""Synchronize release-version metadata across package manifests.

The canonical release version is read from ``pyproject.toml``. In ``--check``
mode the script fails if any supported manifest drifts from that version. In
``--write`` mode it rewrites the manifests to match the canonical version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def canonical_version(root: Path = ROOT) -> str:
    pyproject = load_toml(root / "pyproject.toml")
    return str(pyproject["project"]["version"])


def _replace_r_description_version(text: str, version: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Version: "):
            lines[index] = f"Version: {version}"
            break
    else:  # pragma: no cover - guarded by tests and repository layout
        raise ValueError("Version field not found in R DESCRIPTION")
    return "\n".join(lines) + "\n"


def _replace_first_line(text: str, prefix: str, replacement: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith(prefix):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}{replacement}"
            return "\n".join(lines) + "\n"
    raise ValueError(f"Prefix {prefix!r} not found")


def _replace_julia_project_version(text: str, version: str) -> str:
    return _replace_first_line(text, "version = ", f'version = "{version}"')


def _replace_rust_cargo_version(text: str, version: str) -> str:
    lines = text.splitlines()
    in_package = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if stripped.startswith("[") and stripped.endswith("]") and stripped != "[package]":
            in_package = False
        if in_package and stripped.startswith("version = "):
            lines[index] = f'version = "{version}"'
            return "\n".join(lines) + "\n"
    raise ValueError("Package version field not found in Cargo.toml")


def _replace_csproj_version(text: str, version: str) -> str:
    return _replace_first_line(text, "<Version>", f"<Version>{version}</Version>")


def sync_outputs(root: Path, version: str) -> dict[Path, str]:
    updates: dict[Path, str] = {}

    typescript = json.loads((root / "bindings/typescript/package.json").read_text(encoding="utf-8"))
    typescript["version"] = version
    updates[root / "bindings/typescript/package.json"] = json.dumps(typescript, indent=2) + "\n"

    updates[root / "bindings/julia/Project.toml"] = _replace_julia_project_version(
        (root / "bindings/julia/Project.toml").read_text(encoding="utf-8"),
        version,
    )

    updates[root / "bindings/rust/Cargo.toml"] = _replace_rust_cargo_version(
        (root / "bindings/rust/Cargo.toml").read_text(encoding="utf-8"),
        version,
    )

    r_description = (root / "bindings/r/DESCRIPTION").read_text(encoding="utf-8")
    updates[root / "bindings/r/DESCRIPTION"] = _replace_r_description_version(r_description, version)

    csharp = (root / "bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj").read_text(encoding="utf-8")
    updates[root / "bindings/csharp/Innovate.Kernel/Innovate.Kernel.csproj"] = _replace_csproj_version(csharp, version)

    return updates


def compare_versions(root: Path, version: str) -> list[str]:
    mismatches: list[str] = []
    for path, expected in sync_outputs(root, version).items():
        current = path.read_text(encoding="utf-8")
        if current != expected:
            mismatches.append(str(path.relative_to(root)))
    return mismatches


def apply_versions(root: Path, version: str) -> list[Path]:
    updates = sync_outputs(root, version)
    changed: list[Path] = []
    for path, text in updates.items():
        current = path.read_text(encoding="utf-8")
        if current != text:
            write_text(path, text)
            changed.append(path)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the manifests match the canonical version")
    mode.add_argument("--write", action="store_true", help="rewrite the manifests to match the canonical version")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = canonical_version(args.root)

    if args.write:
        changed = apply_versions(args.root, version)
        if changed:
            print("\n".join(str(path.relative_to(args.root)) for path in changed))
        return 0

    mismatches = compare_versions(args.root, version)
    if mismatches:
        print("Version drift detected:")
        for path in mismatches:
            print(f"- {path}")
        return 1
    print(f"Version manifests are aligned at {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
