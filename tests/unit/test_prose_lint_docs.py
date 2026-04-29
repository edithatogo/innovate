"""Tests for repository prose-lint governance docs."""

from __future__ import annotations

from pathlib import Path


def test_value_prose_lint_style_is_defined() -> None:
    """The repo should define a Vale style for governance prose."""
    style = Path(".vale/styles/Repo/ValueProse.yml")
    assert style.is_file()

    style_text = style.read_text()
    assert "Avoid hedging and filler wording in governance prose." in style_text
    assert "maybe" in style_text
    assert "perhaps" in style_text


def test_governance_docs_reference_value_prose_linting() -> None:
    """The Conductor docs should describe the prose lint policy."""
    product_guidelines = Path("conductor/product-guidelines.md").read_text()
    tech_stack = Path("conductor/tech-stack.md").read_text()
    workflow = Path(".github/workflows/ci.yml").read_text()

    assert "Value Prose Linting" in product_guidelines
    assert "Repo/ValueProse" in product_guidelines
    assert "Vale" in tech_stack
    assert "prose-lint" in workflow
    assert "Run Vale" in workflow
