"""Validate coverage and mutation thresholds are properly configured."""

from __future__ import annotations

from pathlib import Path


def test_coverage_threshold_is_80_in_release_evidence() -> None:
    """COVERAGE_THRESHOLD_LINE_RATE should be 0.80 in release_evidence.py."""
    text = Path("scripts/release_evidence.py").read_text()
    assert "COVERAGE_THRESHOLD_LINE_RATE = 0.80" in text


def test_mutation_threshold_is_70_in_release_evidence() -> None:
    """MUTATION_SCORE_THRESHOLD should be 0.70 in release_evidence.py."""
    text = Path("scripts/release_evidence.py").read_text()
    assert "MUTATION_SCORE_THRESHOLD = 0.70" in text


def test_coverage_session_has_fail_under_80() -> None:
    """Coverage nox session should enforce 80% fail-under threshold."""
    text = Path("noxfile.py").read_text()
    assert "--cov-fail-under=80" in text


def test_coverage_session_has_html_report() -> None:
    """Coverage nox session should produce HTML report."""
    text = Path("noxfile.py").read_text()
    assert "--cov-report=html" in text


def test_coverage_session_has_xml_report() -> None:
    """Coverage nox session should produce XML report."""
    text = Path("noxfile.py").read_text()
    assert "--cov-report=xml" in text


def test_coverage_session_has_threshold_check() -> None:
    """Coverage nox session should enforce COVERAGE_THRESHOLD_LINE_RATE after writing evidence."""
    text = Path("noxfile.py").read_text()
    assert "COVERAGE_THRESHOLD_LINE_RATE" in text
    assert "sys.exit(1)" in text
    # The threshold check should appear near the coverage evidence section
    coverage_section = text.split("def coverage(")[1].split("def ")[0]
    assert "COVERAGE_THRESHOLD_LINE_RATE" in coverage_section
    assert "sys.exit(1)" in coverage_section


def test_mutation_session_has_threshold_check() -> None:
    """Mutation nox session should enforce MUTATION_SCORE_THRESHOLD after writing evidence."""
    text = Path("noxfile.py").read_text()
    assert "MUTATION_SCORE_THRESHOLD" in text
    assert "sys.exit(1)" in text


def test_coverage_threshold_in_pyproject_toml() -> None:
    """pyproject.toml should have fail_under = 80 for coverage."""
    text = Path("pyproject.toml").read_text()
    assert "fail_under = 80" in text


def test_mutation_config_in_pyproject_toml() -> None:
    """pyproject.toml should have mutmut source_paths configured."""
    text = Path("pyproject.toml").read_text()
    assert 'source_paths = ["src/innovate/"]' in text


def test_gate_inventory_lists_correct_thresholds() -> None:
    """Gate inventory should show 80% coverage threshold and >70% mutation threshold."""
    text = Path("conductor/tracks/ci_code_quality_release_hardening_20260625/gate-inventory.md").read_text()
    assert "80% (fail-under=80)" in text
    assert ">70% (enforced)" in text
