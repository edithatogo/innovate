"""Tests for the ecosystem module incubation documentation."""

from __future__ import annotations

from pathlib import Path


def test_ecosystem_module_incubation_docs_are_present() -> None:
    """The ecosystem strategy and contract outline should exist."""
    assert Path("docs/ecosystem/module_incubation_strategy.md").is_file()
    assert Path("specs/ecosystem/README.md").is_file()


def test_ecosystem_module_incubation_docs_define_roles_and_non_goals() -> None:
    """The docs should name ecosystem roles and the intended exclusions."""
    strategy = Path("docs/ecosystem/module_incubation_strategy.md").read_text()
    contract = Path("specs/ecosystem/README.md").read_text()

    for token in ("innovate", "lifecourse", "voiage", "mars", "HEOML"):
        assert token in strategy
        assert token in contract

    assert "Non-Goals" in strategy
    assert "health-economic simulation engines" in strategy
    assert "VOI methods" in strategy
    assert "private sibling-project internals" in strategy

    assert "Non-goals" in contract
    assert "pickle" in contract
    assert "mars core API" in contract
    assert "health-economic simulation or VOI engine" in contract
