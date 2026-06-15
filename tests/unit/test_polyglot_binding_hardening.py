"""Tests for polyglot binding hardening evidence."""

from __future__ import annotations

import json
from pathlib import Path

EVIDENCE_PATH = Path("docs/source/_static/binding_hardening_evidence.json")
INVENTORY_PATH = Path("docs/source/_static/binding_conformance_inventory.json")
SNIPPETS_DIR = Path("docs/source/_static/binding_snippets")


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _inventory_versions() -> dict[str, str]:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    return {entry["language"]: entry["version"] for entry in inventory["bindings"]}


def _assert_hardening_entry(entry: dict[str, object], inventory_versions: dict[str, str]) -> None:
    language = str(entry["language"])
    assert entry["phase"] == "language_binding_hardening"
    assert entry["version"] == inventory_versions[language]
    assert Path(str(entry["package_manifest"])).exists()
    assert entry["package_checks"]
    assert entry["conformance_cases"] >= 6
    assert entry["examples"]
    assert all(Path(str(path)).exists() for path in entry["examples"])
    assert all(Path(str(path)).exists() for path in entry["source_paths"])
    assert Path(str(entry["snippet"])).exists()


def test_r_julia_typescript_hardening_evidence_exists() -> None:
    """R, Julia, and TypeScript bindings should have package-check evidence."""
    evidence = {entry["language"]: entry for entry in _evidence()["bindings"]}
    inventory_versions = _inventory_versions()

    for language in ["r", "julia", "typescript"]:
        _assert_hardening_entry(evidence[language], inventory_versions)


def test_r_julia_typescript_docs_snippets_are_language_specific() -> None:
    """Docs snippets should be idiomatic and language-specific."""
    expected = {
        "r": ["library(innovate.R)", "predict_model"],
        "julia": ["using Innovate", "predict_model"],
        "typescript": ["import", "predictModel"],
    }

    for language, markers in expected.items():
        text = (SNIPPETS_DIR / f"{language}.md").read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text


def test_go_csharp_rust_hardening_evidence_exists() -> None:
    """Go, C#, and Rust bindings should have package-check evidence."""
    evidence = {entry["language"]: entry for entry in _evidence()["bindings"]}
    inventory_versions = _inventory_versions()

    for language in ["go", "csharp", "rust"]:
        _assert_hardening_entry(evidence[language], inventory_versions)


def test_go_csharp_rust_docs_snippets_are_language_specific() -> None:
    """Compiled bindings should expose idiomatic predict examples."""
    expected = {
        "go": ["package main", "PredictModel"],
        "csharp": ["using Innovate.Kernel", "PredictModelAsync"],
        "rust": ["use innovate", "predict_model"],
    }

    for language, markers in expected.items():
        text = (SNIPPETS_DIR / f"{language}.md").read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text
