"""Tests for polyglot binding hardening evidence."""

from __future__ import annotations

import json
from pathlib import Path

EVIDENCE_PATH = Path("docs/source/_static/binding_hardening_evidence.json")
SNIPPETS_DIR = Path("docs/source/_static/binding_snippets")


def _evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_r_julia_typescript_hardening_evidence_exists() -> None:
    """R, Julia, and TypeScript bindings should have package-check evidence."""
    evidence = {entry["language"]: entry for entry in _evidence()["bindings"]}

    for language in ["r", "julia", "typescript"]:
        entry = evidence[language]
        assert entry["phase"] == "language_binding_hardening"
        assert entry["package_checks"]
        assert entry["conformance_cases"] >= 6
        assert entry["examples"]
        assert all(Path(path).exists() for path in entry["examples"])
        assert all(Path(path).exists() for path in entry["source_paths"])


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

    for language in ["go", "csharp", "rust"]:
        entry = evidence[language]
        assert entry["phase"] == "language_binding_hardening"
        assert entry["package_checks"]
        assert entry["conformance_cases"] >= 6
        assert entry["examples"]
        assert all(Path(path).exists() for path in entry["examples"])
        assert all(Path(path).exists() for path in entry["source_paths"])


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
