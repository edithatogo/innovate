"""Local selection must fail closed on changes outside Python tracing."""

import importlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# The pytest console entrypoint does not add the repository root to sys.path.
with pytest.MonkeyPatch.context() as import_path:
    import_path.syspath_prepend(str(REPO))
    profiles = importlib.import_module("scripts.test_acceleration.profiles")


@pytest.fixture
def source_tree(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name, value in {
        ".gitignore": ".pytest_cache/\n.venv/\n",
        "pyproject.toml": "[project]\nname = 'synthetic'\n",
        "uv.lock": "version = 1\n",
        "Cargo.lock": "version = 4\n",
        "rust/src/lib.rs": "pub fn value() -> u8 { 1 }\n",
        "bindings/python/innovate/__init__.py": "def value():\n    return 1\n",
        "scripts/helper.py": "print('synthetic')\n",
        "tests/native/test_example.py": "def test_example():\n    assert True\n",
        "specs/input.json": '{"value": 1}\n',
        "migration-ledger.json": '{"release_ready": false}\n',
        "conductor/evidence/receipt.json": '{"source": "synthetic"}\n',
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def snapshot(root, **kwargs):
    return profiles.snapshot(root, "native", {"python": "3.15.2", "pytest": "9.2.0"}, **kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "uv.lock",
        "Cargo.lock",
        "rust/src/lib.rs",
        "scripts/helper.py",
        "specs/input.json",
        "migration-ledger.json",
        "conductor/evidence/receipt.json",
    ],
)
def test_untraced_changes_invalidate_the_database(source_tree, name):
    before = snapshot(source_tree)
    path = source_tree / name
    path.write_text(path.read_text() + "\nchanged\n")
    assert profiles.invalidation_reasons(before, snapshot(source_tree))


def test_python_changes_remain_eligible_for_dependency_selection(source_tree):
    before = snapshot(source_tree)
    path = source_tree / "bindings/python/innovate/__init__.py"
    path.write_text("def value():\n    return 2\n")
    assert profiles.invalidation_reasons(before, snapshot(source_tree)) == []


def test_test_inventory_addition_and_deletion_require_full_rebuild(source_tree):
    before = snapshot(source_tree)
    new = source_tree / "tests/native/test_new.py"
    new.write_text("def test_new():\n    assert True\n")
    assert "inventory" in profiles.invalidation_reasons(before, snapshot(source_tree))
    new.unlink()
    (source_tree / "tests/native/test_example.py").unlink()
    assert "inventory" in profiles.invalidation_reasons(before, snapshot(source_tree))


def test_config_and_dependency_changes_require_rebuild(source_tree):
    before = snapshot(source_tree)
    changed = profiles.snapshot(source_tree, "native", {"python": "3.15.2", "pytest": "9.3.0"})
    assert "environment" in profiles.invalidation_reasons(before, changed)
    (source_tree / "tests/conftest.py").write_text("VALUE = 1\n")
    seeded = snapshot(source_tree)
    (source_tree / "tests/conftest.py").write_text("VALUE = 2\n")
    assert profiles.invalidation_reasons(seeded, snapshot(source_tree))


def test_rebuilt_native_binary_invalidates_even_with_unchanged_source(source_tree):
    binary = source_tree / ".venv/native.so"
    binary.parent.mkdir()
    binary.write_bytes(b"synthetic binary version one")
    before = snapshot(source_tree, native_paths=[binary])
    binary.write_bytes(b"synthetic binary version two")
    assert "native" in profiles.invalidation_reasons(before, snapshot(source_tree, native_paths=[binary]))


def test_cache_receipts_do_not_invalidate_themselves(source_tree):
    before = snapshot(source_tree)
    cache = source_tree / ".pytest_cache/test_acceleration/runs"
    cache.mkdir(parents=True)
    (cache / "receipt.json").write_text(json.dumps({"wall_seconds": 0.25}))
    assert snapshot(source_tree) == before


@pytest.mark.parametrize(
    "name",
    [
        "CI",
        "GITHUB_ACTIONS",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "COVERAGE_PROCESS_START",
        "COVERAGE_CORE",
        "TESTMON_DATAFILE",
    ],
)
def test_incompatible_environment_is_rejected(name):
    with pytest.raises(profiles.ProfileError):
        profiles.validate_environment({name: "true"}, (3, 15, 2))


def test_unsupported_interpreter_is_rejected():
    with pytest.raises(profiles.ProfileError, match="3.15.2"):
        profiles.validate_environment({}, (3, 13, 5))


def test_serial_cache_lease_does_not_steal_an_active_writer(tmp_path):
    with profiles.cache_lease(tmp_path):
        with pytest.raises(profiles.ProfileError, match="already in use"):
            with profiles.cache_lease(tmp_path):
                pytest.fail("a second writer acquired the cache")
    with profiles.cache_lease(tmp_path):
        pass


def test_only_a_seeded_warm_testmon_run_may_select_nothing():
    metrics = {
        "collected": 0,
        "selected": 0,
        "executed": 0,
        "errors": 0,
        "failed": 0,
        "skipped": 0,
        "exitstatus": 5,
        "passed": 0,
    }
    assert profiles.classify_result("testmon", 5, metrics, cached_items=16, rebuilt=False) == "no_tests_selected"
    for mode, cached, rebuilt in [("baseline", 16, False), ("testmon", 0, False), ("testmon", 16, True)]:
        assert profiles.classify_result(mode, 5, metrics, cached_items=cached, rebuilt=rebuilt) == "failed"


def test_no_receipt_or_collection_error_cannot_pass():
    assert profiles.classify_result("baseline", 0, None) == "failed"
    metrics = {"collected": 1, "selected": 1, "executed": 0, "errors": 1, "failed": 0, "skipped": 0, "exitstatus": 2}
    assert profiles.classify_result("testmon", 2, metrics, cached_items=16) == "failed"


def test_all_call_skips_cannot_seed_a_successful_baseline():
    metrics = {
        "collected": 1,
        "selected": 1,
        "executed": 1,
        "errors": 0,
        "failed": 0,
        "skipped": 1,
        "passed": 0,
        "exitstatus": 0,
    }
    assert profiles.classify_result("testmon", 0, metrics, rebuilt=True) == "failed"
    assert profiles.classify_result("baseline", 0, metrics) == "failed"


@pytest.mark.parametrize("metrics", [{}, {"errors": 0}, {"passed": True}])
def test_invalid_metrics_fail_closed(metrics):
    assert profiles.classify_result("testmon", 0, metrics, cached_items=16) == "failed"
