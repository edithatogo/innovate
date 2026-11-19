"""An installed-wheel smoke receipt preserves the actual test outcomes."""

import hashlib
import json
import sys
from importlib.machinery import ExtensionFileLoader
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree

import pytest
from scripts import record_release_wheel_evidence as recorder
from scripts.record_release_wheel_evidence import LIVE_OECD, SMOKE_MODULES, smoke_results


def smoke_report():
    root = ElementTree.Element("testsuite")
    for module in sorted(SMOKE_MODULES):
        ElementTree.SubElement(root, "testcase", classname=f"tests.native.{module}", name="test_example")
    module, name = LIVE_OECD
    skipped = ElementTree.SubElement(root, "testcase", classname=f"tests.native.{module}", name=name)
    ElementTree.SubElement(skipped, "skipped")
    return root


def test_smoke_receipt_records_optional_oecd_skip_separately():
    passed, skipped = smoke_results(ElementTree.tostring(smoke_report()))
    assert passed == dict.fromkeys(SMOKE_MODULES, 1)
    assert skipped == [".".join(LIVE_OECD)]


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
def test_smoke_receipt_rejects_missing_or_unsuccessful_required_cases(change, diagnostic):
    root = smoke_report()
    if change == "duplicate":
        root.append(root[0])
    elif change == "omitted":
        root.remove(root[0])
    else:
        ElementTree.SubElement(root[0], change)
    with pytest.raises(ValueError, match=diagnostic):
        smoke_results(ElementTree.tostring(root))


@pytest.fixture
def installed_case(tmp_path, monkeypatch):
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
    report.write_bytes(ElementTree.tostring(smoke_report()))

    def verify(candidate, directory, imported):
        assert candidate == wheel and directory == package
        assert imported == [package / "__init__.py", native_file]
        return {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "verified_package_members": 2}

    monkeypatch.setattr(recorder, "verify_wheel", verify)
    return environment, wheel, report, tmp_path / "evidence.json"


def test_installed_receipt_binds_source_artifact_and_observed_results(installed_case, monkeypatch):
    environment, wheel, report, output = installed_case
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
    assert receipt["artifact_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert receipt["pytest_report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert receipt["passed_by_module"] == dict.fromkeys(SMOKE_MODULES, 1)
    assert receipt["skipped"] == [".".join(LIVE_OECD)]


def test_stale_wheel_source_cannot_produce_evidence(installed_case):
    _, wheel, _, output = installed_case
    (wheel.parent / "SOURCE_SHA256").write_text("c" * 64, encoding="utf-8")
    with pytest.raises(ValueError, match="different source inputs"):
        recorder.record(*installed_case)
    assert not output.exists()
