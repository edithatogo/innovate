"""Validate native development versions without advertising removed bindings."""

import argparse
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    del args
    python = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    rust = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    julia = tomllib.loads((ROOT / "bindings/julia/Project.toml").read_text())["version"]
    r = next(
        line.removeprefix("Version: ")
        for line in (ROOT / "bindings/r/DESCRIPTION").read_text().splitlines()
        if line.startswith("Version: ")
    )
    if (python, rust, julia, r) != ("0.7.0a1", "0.7.0-alpha.1", "0.7.0-DEV", "0.7.0.9000"):
        raise SystemExit("Native development version mapping drifted; qualify a new release mapping explicitly")
    print("Native development version mapping valid; release remains gated")


if __name__ == "__main__":
    main()
