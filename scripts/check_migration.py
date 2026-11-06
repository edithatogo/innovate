"""Fail closed on feature loss and native migration release claims."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if sys.version_info[:3] != (3, 15, 2):
        raise ValueError("Use the selected repo-local Python 3.15.2 interpreter")
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    ledger = json.loads((ROOT / "migration-ledger.json").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    if project["build-system"]["build-backend"] != "maturin":
        raise ValueError("Native Maturin build backend is required")
    if project["project"]["dependencies"] != ["polars==2.1.0"]:
        raise ValueError("Polars is the only base Python runtime dependency")
    expected_models = {
        "bass",
        "complementary_goods",
        "composite",
        "fisher_pry",
        "gompertz",
        "hierarchical",
        "latent_process",
        "lock_in",
        "logistic",
        "lotka_volterra",
        "mixture",
        "multi_product",
        "network_diffusion",
        "norton_bass",
        "policy_hazard",
        "regime_switching",
    }
    if {m["key"] for m in ledger["models"]} != expected_models:
        raise ValueError("Migration ledger lost a required scientific model")
    pending = [
        item["key"]
        for group in ("models", "feature_areas")
        for item in ledger[group]
        if item["required"] and item["status"] != "qualified"
    ]
    if args.release and (pending or not ledger["release_ready"]):
        print("Release blocked: scientific migration and acceptance remain incomplete:", ", ".join(pending))
        return 1
    print(f"Native development policy valid; {len(pending)} feature obligations still block release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
