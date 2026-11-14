"""Opt-in executable qualification of real testmon selection and recovery.

Run this file explicitly with the selected Python and optional pinned profile
dependencies installed. It is outside the ordinary ``tests/native`` CI scope.
Missing or incompatible tools fail; they never produce a capability skip.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from scripts.test_acceleration import profiles

REPO = Path(__file__).resolve().parents[2]


def test_real_testmon_selection_failure_and_opaque_input_recovery(tmp_path, monkeypatch):
    """Select a changed binding function, then reseed after failures or opaque edits."""
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    files = {
        ".gitignore": ".pytest_cache/\n__pycache__/\n*.pyc\n",
        "pyproject.toml": '[tool.pytest.ini_options]\nminversion = "9.2"\ntestpaths = ["tests/native"]\n',
        "bindings/__init__.py": "",
        "bindings/python/__init__.py": "",
        "bindings/python/fixture_binding.py": "def first():\n    return 2\n\ndef second():\n    return 20\n",
        "tests/native/test_binding.py": (
            "from bindings.python.fixture_binding import first, second\n\n"
            "def test_first():\n    assert first() == 2\n\n"
            "def test_second():\n    assert second() == 20\n"
        ),
        "rust/src/lib.rs": "pub fn synthetic() -> u8 { 1 }\n",
        "specs/fixture.json": '{"synthetic_value": 1}\n',
    }
    for name, value in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    for name in (
        "scripts/__init__.py",
        "scripts/test_acceleration/__init__.py",
        "scripts/test_acceleration/metrics.py",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, path)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    monkeypatch.setattr(profiles, "ROOT", source)
    receipts = []

    def run(label, mode, status, executed, reason=None):
        try:
            receipt, code = profiles.run_profile(mode, "native", 5, False)
        except profiles.ProfileError as exc:
            pytest.fail(
                f"Local testmon qualification refused: {exc}. Use Python 3.15.2 and install "
                "scripts/test_acceleration/requirements.txt with uv pip install --python <selected-python> "
                "--require-hashes -r scripts/test_acceleration/requirements.txt.",
                pytrace=False,
            )
        receipt["case"] = label
        receipts.append(receipt)
        (tmp_path / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
        assert receipt["status"] == status, receipt
        assert code == (1 if status == "failed" else 0), receipt
        assert receipt["pytest"]["executed"] == executed, receipt
        assert receipt["pytest"]["errors"] == receipt["pytest"]["skipped"] == 0, receipt
        assert receipt["promotion_evidence"] is False
        assert (source / receipt["local_receipt"]).is_file()
        if reason:
            assert reason in receipt["invalidation_reasons"], receipt
        else:
            assert receipt["invalidation_reasons"] == [], receipt
        return receipt

    baseline = run("baseline", "baseline", "passed", 2)
    assert baseline["pytest"]["collected"] == baseline["pytest"]["passed"] == 2
    cold = run("testmon_cold", "testmon", "passed", 2, "unseeded")
    assert cold["pytest"]["collected"] == cold["pytest"]["passed"] == 2
    warm = run("testmon_warm", "testmon", "no_tests_selected", 0)
    assert warm["pytest"]["collected"] == warm["pytest"]["passed"] == 0
    assert warm["seed_inventory_count"] == 2

    # Change the body size too, avoiding timestamp/size-based .pyc reuse. Only
    # eligible Python code changes: testmon itself must catch the failing test.
    binding = source / "bindings/python/fixture_binding.py"
    binding.write_text("def first():\n    return 999\n\ndef second():\n    return 20\n")
    changed = run("changed_binding_failure", "testmon", "failed", 1)
    assert changed["pytest"]["collected"] == 2
    assert changed["pytest"]["selected"] == changed["pytest"]["failed"] == changed["pytest"]["deselected"] == 1
    state = source / ".pytest_cache/test_acceleration/testmon-native/state.json"
    assert not state.exists()

    binding.write_text(files["bindings/python/fixture_binding.py"])
    recovered = run("failure_recovery_reseeds", "testmon", "passed", 2, "unseeded")
    assert recovered["pytest"]["selected"] == recovered["pytest"]["passed"] == 2
    assert json.loads(state.read_text())["inventory_count"] == 2

    (source / "rust/src/lib.rs").write_text("pub fn synthetic() -> u8 { 2 }\n")
    run("native_source_change", "testmon", "passed", 2, "opaque")
    (source / "specs/fixture.json").write_text('{"synthetic_value": 2}\n')
    run("fixture_change", "testmon", "passed", 2, "opaque")
    (source / "pyproject.toml").write_text(files["pyproject.toml"] + "# synthetic manifest revision\n")
    run("manifest_change", "testmon", "passed", 2, "opaque")
    (source / "tests/native/test_added.py").write_text("def test_added():\n    assert 1 + 1 == 2\n")
    added = run("added_test", "testmon", "passed", 3, "inventory")
    assert added["pytest"]["collected"] == added["pytest"]["passed"] == 3
    final = run("final_warm", "testmon", "no_tests_selected", 0)
    assert final["seed_inventory_count"] == 3
