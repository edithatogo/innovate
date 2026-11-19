"""An installed-wheel smoke receipt preserves the actual test outcomes."""

import hashlib
import json
import sys
from importlib import import_module
from importlib.machinery import ExtensionFileLoader
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree

import pytest


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    return import_module("scripts.record_release_wheel_evidence")


def smoke_report(recorder):
    root = ElementTree.Element("testsuite")
    for module in sorted(recorder.SMOKE_MODULES):
        ElementTree.SubElement(root, "testcase", classname=f"tests.native.{module}", name="test_example")
    for name in sorted(recorder.ORACLE_CASES):
        ElementTree.SubElement(root, "testcase", classname=f"tests.validation.{recorder.ORACLE_MODULE}", name=name)
    module, name = recorder.LIVE_OECD
    skipped = ElementTree.SubElement(root, "testcase", classname=f"tests.native.{module}", name=name)
    ElementTree.SubElement(skipped, "skipped")
    return root


def smoke_counts(recorder):
    return dict.fromkeys(recorder.SMOKE_MODULES, 1) | {recorder.ORACLE_MODULE: len(recorder.ORACLE_CASES) + 1}


def test_smoke_receipt_records_optional_oecd_skip_separately(recorder):
    passed, skipped = recorder.smoke_results(ElementTree.tostring(smoke_report(recorder)))
    assert passed == smoke_counts(recorder)
    assert skipped == [".".join(recorder.LIVE_OECD)]


def test_smoke_receipt_requires_each_named_analytic_prediction(recorder):
    root = smoke_report(recorder)
    oracle = next(case for case in root if case.get("name") in recorder.ORACLE_CASES)
    root.remove(oracle)
    with pytest.raises(ValueError, match="analytic model suite is incomplete"):
        recorder.smoke_results(ElementTree.tostring(root))


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("duplicate", "repeated installed-wheel case"),
        ("omitted", "suite is incomplete"),
        ("failure", "case failed"),
        ("error", "case failed"),
        ("skipped", "unexpectedly skipped"),
    ],
)
def test_smoke_receipt_rejects_missing_or_unsuccessful_required_cases(recorder, change, diagnostic):
    root = smoke_report(recorder)
    if change == "duplicate":
        root.append(root[0])
    elif change == "omitted":
        root.remove(root[0])
    else:
        ElementTree.SubElement(root[0], change)
    with pytest.raises(ValueError, match=diagnostic):
        recorder.smoke_results(ElementTree.tostring(root))


@pytest.fixture
def installed_case(recorder, tmp_path, monkeypatch):
    environment = tmp_path / "venv"
    package = environment / "site-packages/innovate"
    package.mkdir(parents=True)
    module = ModuleType("innovate")
    module.__file__ = str(package / "__init__.py")
    native_file = package / "_native.so"
    native_file.touch()
    (package / "__init__.py").touch()
    module._native = SimpleNamespace(
        __file__=str(native_file), __loader__=ExtensionFileLoader("innovate._native", str(native_file))
    )
    monkeypatch.setitem(sys.modules, "innovate", module)
    monkeypatch.setattr(
        recorder,
        "platform",
        SimpleNamespace(system=lambda: "Linux", machine=lambda: "x86_64", python_version=lambda: "3.15.2"),
    )
    monkeypatch.setattr(recorder, "subprocess", SimpleNamespace(check_output=lambda *_args, **_kwargs: "a" * 40))
    monkeypatch.setattr(recorder, "release_source_digest", lambda: "b" * 64)
    (tmp_path / "SOURCE_SHA256").write_text("b" * 64 + "\n", encoding="utf-8")
    wheel = tmp_path / "innovate.whl"
    wheel.write_bytes(b"synthetic fixture")
    report = tmp_path / "report.xml"
    report.write_bytes(ElementTree.tostring(smoke_report(recorder)))

    def verify(candidate, directory, imported):
        assert candidate == wheel and directory == package
        assert imported == [package / "__init__.py", native_file]
        return {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "verified_package_members": 2}

    monkeypatch.setattr(recorder, "verify_wheel", verify)
    return environment, wheel, report, tmp_path / "evidence.json"


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    [
        ("Linux", "x86_64", "linux-x86_64"),
        ("Darwin", "arm64", "macos-arm64"),
        ("Windows", "AMD64", "windows-x86_64"),
    ],
)
def test_installed_receipt_binds_source_artifact_and_observed_results(
    recorder, installed_case, monkeypatch, system, machine, target
):
    environment, wheel, report, output = installed_case
    monkeypatch.setattr(recorder.platform, "system", lambda: system)
    monkeypatch.setattr(recorder.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_release_wheel_evidence",
            "--environment",
            str(environment),
            "--wheel",
            str(wheel),
            "--report",
            str(report),
            "--output",
            str(output),
        ],
    )
    assert recorder.main() == 0
    receipt = json.loads(output.read_bytes())
    assert receipt["kind"] == "bounded_installed_execution"
    assert receipt["source_commit"] == "a" * 40
    assert receipt["source_sha256"] == "b" * 64
    assert receipt["target"] == target
    assert receipt["artifact_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert receipt["pytest_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert receipt["passed_by_module"] == smoke_counts(recorder)
    assert receipt["skipped"] == [".".join(recorder.LIVE_OECD)]


def test_stale_wheel_source_cannot_produce_evidence(recorder, installed_case):
    _, wheel, _, output = installed_case
    (wheel.parent / "SOURCE_SHA256").write_text("c" * 64, encoding="utf-8")
    with pytest.raises(ValueError, match="different source inputs"):
        recorder.record(*installed_case)
    assert not output.exists()


def test_unqualified_host_cannot_produce_evidence(recorder, installed_case, monkeypatch):
    monkeypatch.setattr(recorder.platform, "machine", lambda: "aarch64")
    with pytest.raises(ValueError, match="supported release host"):
        recorder.record(*installed_case)
    assert not installed_case[-1].exists()
