"""Incomplete native support cannot silently retire an existing feature."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("mutation", ["drop_feature", "drop_model", "make_optional", "duplicate_feature"])
def test_migration_policy_rejects_feature_loss(tmp_path, monkeypatch, mutation):
    spec = importlib.util.spec_from_file_location("migration_policy", ROOT / "scripts/check_migration.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ledger = json.loads((ROOT / "migration-ledger.json").read_text())
    if mutation == "drop_feature":
        ledger["feature_areas"].pop()
    elif mutation == "drop_model":
        ledger["models"].pop()
    elif mutation == "make_optional":
        ledger["feature_areas"][0]["required"] = False
    else:
        ledger["feature_areas"].append(ledger["feature_areas"][0])
    (tmp_path / "migration-ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "pyproject.toml").write_text((ROOT / "pyproject.toml").read_text())
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["check_migration.py"])
    with pytest.raises(ValueError, match="(lost|required)"):
        module.main()
