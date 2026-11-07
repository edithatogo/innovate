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
    expected_features = {
        "covariates_and_events",
        "multi_generation_substitution",
        "competitive_ecosystems",
        "hierarchical_mixture_inference",
        "latent_regime_models",
        "network_policy_diffusion",
        "kairos_des_abm",
        "bayesian_inference",
        "bootstrap_uncertainty",
        "batch_fitting",
        "preprocessing",
        "causal_analysis",
        "adopter_classification",
        "data_ingestion_provenance_validation",
        "scenario_registry_execution",
        "explainability_sensitivity_reporting",
        "benchmark_model_cards",
        "plugin_contracts",
        "remote_execution",
        "accelerator_hpc",
        "voiage_arrow_handoff",
        "calibration_identifiability",
    }
    if {f["key"] for f in ledger["feature_areas"]} != expected_features:
        raise ValueError("Migration ledger lost a required current or planned feature area")
    for group in ("models", "feature_areas"):
        items = ledger[group]
        if len(items) != len({item["key"] for item in items}) or any(item["required"] is not True for item in items):
            raise ValueError("Every unique model and feature obligation must remain required")
    for group in ("models", "feature_areas"):
        for item in ledger[group]:
            status = item.get("status")
            if status not in ("pending", "bounded_native", "qualified"):
                raise ValueError(f"Invalid migration status for {item['key']}")
            evidence = item.get("evidence")
            if evidence is None and status == "pending":
                continue
            if not isinstance(evidence, str) or not evidence:
                raise ValueError(f"Missing migration evidence for {item['key']}")
            evidence_path = Path(evidence)
            resolved = (ROOT / evidence_path).resolve()
            if evidence_path.is_absolute() or not resolved.is_relative_to(ROOT.resolve()) or not resolved.is_file():
                raise ValueError(f"Invalid migration evidence for {item['key']}: {evidence}")

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
