"""Reject incomplete installed-Arrow evidence before it can qualify a candidate."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest


@pytest.fixture
def qualifier() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts/qualify_scenario_arrow.py"
    spec = importlib.util.spec_from_file_location("scenario_arrow_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_versions() -> dict[str, str]:
    return {"polars": "2.1.0", "pyarrow": "26.0.1", "pytest": "9.2.0"}


def build_record(inputs: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "built",
        "build_exit_code": 0,
        "wheel_sha256": "wheel-digest",
        "source_inputs": inputs,
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
        "kairos_revision": "c" * 40,
        "build_log_sha256": "d" * 64,
        "rustc": "fixture compiler",
        "cargo": "fixture cargo",
        "command": "fixture build",
        "build_profile": "dev",
        "build_log": "build.log",
        "build_jobs": 1,
        "worktree_changes": [],
    }


def junit(qualifier: ModuleType) -> ET.Element:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for module, name in sorted(qualifier.EXPECTED_CASES):
        ET.SubElement(suite, "testcase", classname=f"tests.native.{module}", name=name)
    return root


def make_wheel(tmp_path: Path, *, bad_record: bool = False, missing_native: bool = False) -> tuple[Path, Path]:
    package = tmp_path / "site-packages" / "innovate"
    package.mkdir(parents=True)
    members = {"innovate/__init__.py": b"version = 'fixture'\n"}
    if not missing_native:
        members["innovate/_native.fixture.so"] = b"synthetic module bytes; never executed"
    wheel = tmp_path / "innovate-0.7.0a1-fixture.whl"
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    with ZipFile(wheel, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
            (package / Path(name).name).write_bytes(data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([name, "sha256=" + ("wrong" if bad_record else digest), str(len(data))])
        writer.writerow(["innovate-0.7.0a1.dist-info/RECORD", "", ""])
        archive.writestr("innovate-0.7.0a1.dist-info/RECORD", record.getvalue())
    return wheel, package


def test_declared_installed_versions_are_accepted(qualifier):
    qualifier.validate_versions((3, 15, 2), runtime_versions(), runtime_versions())


@pytest.mark.parametrize("missing", ["polars", "pyarrow", "pytest"])
def test_missing_dependency_cannot_be_treated_as_skip(qualifier, missing):
    installed = runtime_versions()
    del installed[missing]
    with pytest.raises(ValueError, match=missing):
        qualifier.validate_versions((3, 15, 2), installed, runtime_versions())


@pytest.mark.parametrize("mismatch", ["polars", "pyarrow", "pytest"])
def test_changed_dependency_version_is_rejected(qualifier, mismatch):
    installed = runtime_versions()
    installed[mismatch] = "0.0.0"
    with pytest.raises(ValueError, match=mismatch):
        qualifier.validate_versions((3, 15, 2), installed, runtime_versions())


def test_wrong_python_is_rejected(qualifier):
    with pytest.raises(ValueError, match="Python"):
        qualifier.validate_versions((3, 15, 1), runtime_versions(), runtime_versions())


def test_python_fallback_cannot_be_reported_as_native(qualifier):
    loader = importlib.machinery.SourceFileLoader("innovate._native", "/synthetic/_native.py")
    with pytest.raises(ValueError, match="compiled extension"):
        qualifier.validate_native_loader(loader)


def test_compiled_extension_loader_is_accepted(qualifier):
    qualifier.validate_native_loader(
        importlib.machinery.ExtensionFileLoader("innovate._native", "/synthetic/_native.so")
    )


def test_all_named_export_and_producer_cases_are_required(qualifier):
    result = qualifier.validate_junit(junit(qualifier))
    assert result == {"passed": 42, "exports": 38, "optional_arrow": 4, "native_producers": 4}


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_any_unsuccessful_case_rejects_qualification(qualifier, outcome):
    tree = junit(qualifier)
    case = next(c for c in tree.iter("testcase") if "optional_pyarrow" in c.attrib["name"])
    ET.SubElement(case, outcome, message="synthetic negative control")
    with pytest.raises(ValueError, match=outcome):
        qualifier.validate_junit(tree)


def test_missing_optional_parameter_case_is_rejected(qualifier):
    tree = junit(qualifier)
    suite = tree.find("testsuite")
    case = next(c for c in suite if "optional_pyarrow" in c.attrib["name"])
    suite.remove(case)
    with pytest.raises(ValueError, match="missing"):
        qualifier.validate_junit(tree)


def test_duplicate_case_cannot_replace_missing_case(qualifier):
    tree = junit(qualifier)
    cases = list(tree.iter("testcase"))
    cases[-1].attrib.update(cases[0].attrib)
    with pytest.raises(ValueError, match="duplicate"):
        qualifier.validate_junit(tree)


def test_extra_or_empty_selection_is_not_the_qualified_scope(qualifier):
    tree = junit(qualifier)
    ET.SubElement(tree.find("testsuite"), "testcase", classname="unrelated", name="test_not_in_scope")
    with pytest.raises(ValueError, match="unexpected"):
        qualifier.validate_junit(tree)
    with pytest.raises(ValueError, match="missing"):
        qualifier.validate_junit(ET.Element("testsuites"))


def test_supplied_wheel_matches_installed_bytes_and_record(qualifier, tmp_path):
    wheel, package = make_wheel(tmp_path)
    result = qualifier.verify_wheel(wheel, package, [package / "__init__.py", package / "_native.fixture.so"])
    assert result["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert result["verified_package_members"] == 2


def test_different_installed_native_library_is_rejected(qualifier, tmp_path):
    wheel, package = make_wheel(tmp_path)
    (package / "_native.fixture.so").write_bytes(b"different artifact")
    with pytest.raises(ValueError, match="installed.*differs"):
        qualifier.verify_wheel(wheel, package, [package / "__init__.py", package / "_native.fixture.so"])


def test_invalid_record_hash_is_rejected(qualifier, tmp_path):
    wheel, package = make_wheel(tmp_path, bad_record=True)
    with pytest.raises(ValueError, match="RECORD"):
        qualifier.verify_wheel(wheel, package, [package / "__init__.py", package / "_native.fixture.so"])


def test_missing_native_member_is_rejected(qualifier, tmp_path):
    wheel, package = make_wheel(tmp_path, missing_native=True)
    with pytest.raises(ValueError, match="required"):
        qualifier.verify_wheel(wheel, package, [package / "__init__.py", package / "_native.fixture.so"])


def test_wheel_traversal_is_rejected(qualifier, tmp_path):
    wheel, package = make_wheel(tmp_path)
    with ZipFile(wheel, "a") as archive:
        archive.writestr("innovate/../outside", b"never extract this")
    with pytest.raises(ValueError, match="unsafe"):
        qualifier.verify_wheel(wheel, package, [package / "__init__.py", package / "_native.fixture.so"])


def test_test_source_drift_rejects_qualification(qualifier, tmp_path):
    source = tmp_path / "test_source.py"
    source.write_bytes(b"original")
    identities = {"test_source.py": hashlib.sha256(b"original").hexdigest()}
    qualifier.verify_input_hashes(tmp_path, identities)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="test_source.py"):
        qualifier.verify_input_hashes(tmp_path, identities)


def test_successful_build_record_binds_exact_inputs_and_wheel(qualifier):
    inputs = {"Cargo.lock": "lock-digest", "rust/source.rs": "source-digest"}
    record = build_record(inputs)
    qualifier.validate_build_record(record, inputs, "wheel-digest")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema_version"),
        ("status", "building", "completed build"),
        ("build_exit_code", 1, "completed build"),
        ("build_exit_code", False, "completed build"),
        ("wheel_sha256", "different-wheel", "wheel hash"),
        ("source_inputs", {"Cargo.lock": "different-lock"}, "source inputs"),
    ],
)
def test_failed_or_mismatched_build_record_is_rejected(qualifier, field, value, message):
    inputs = {"Cargo.lock": "lock-digest"}
    record = build_record(inputs)
    record[field] = value
    with pytest.raises(ValueError, match=message):
        qualifier.validate_build_record(record, inputs, "wheel-digest")


@pytest.mark.parametrize("record", [[], None, "not an object", 1, False])
def test_non_object_build_record_is_rejected(qualifier, record):
    with pytest.raises(ValueError, match="object"):
        qualifier.validate_build_record(record, {}, "wheel-digest")


@pytest.mark.parametrize(
    "field",
    [
        "source_commit",
        "source_tree",
        "kairos_revision",
        "rustc",
        "cargo",
        "command",
        "build_log",
        "build_log_sha256",
        "build_jobs",
        "worktree_changes",
    ],
)
def test_missing_build_identity_is_rejected(qualifier, field):
    inputs = {"Cargo.lock": "lock-digest"}
    record = build_record(inputs)
    del record[field]
    with pytest.raises(ValueError, match=field):
        qualifier.validate_build_record(record, inputs, "wheel-digest")
