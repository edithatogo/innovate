"""Tests for the XLA backend strategy documentation."""

from __future__ import annotations

from pathlib import Path


DOC_PATH = Path("docs/astro-site/src/content/docs/operations/xla-backend.md")
LATEST_DOC_PATH = Path("docs/astro-site/src/content/docs/latest/operations/xla-backend.md")
STARLIGHT_CONFIG = Path("docs/astro-site/starlight.config.mjs")


def _strategy_text() -> str:
    return "\n".join((DOC_PATH.read_text(), LATEST_DOC_PATH.read_text()))


def test_xla_backend_strategy_doc_exists_and_is_linked() -> None:
    """The XLA policy should be a first-class Starlight document."""
    starlight_config = STARLIGHT_CONFIG.read_text()
    principles = Path("docs/architecture_principles.md").read_text()
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()

    assert DOC_PATH.is_file()
    assert LATEST_DOC_PATH.is_file()
    assert "/operations/xla-backend/" in starlight_config
    assert "slug: latest/operations/xla-backend" in LATEST_DOC_PATH.read_text()
    assert "XLA Backend Strategy" in principles
    assert "docs/astro-site/src/content/docs/operations/xla-backend.md" in roadmap


def test_xla_backend_strategy_defines_eligibility_and_rejection_rules() -> None:
    """The strategy should say when XLA is appropriate and when to reject it."""
    strategy = _strategy_text()

    for phrase in (
        "XLA-eligible",
        "static",
        "jax.lax.scan",
        "jax.lax.while_loop",
        "explicit JAX PRNG keys",
        "event queues",
        "unbounded shape changes",
        "discrete-event simulation",
    ):
        assert phrase in strategy


def test_xla_backend_strategy_names_preferred_libraries() -> None:
    """The policy should identify the preferred XLA-aligned libraries."""
    strategy = _strategy_text()

    for library in (
        "JAX",
        "NumPyro",
        "BlackJAX",
        "TensorFlow Probability's JAX substrate",
        "Diffrax",
        "NumPy/SciPy remains the reference path",
    ):
        assert library in strategy


def test_xla_backend_strategy_defines_promotion_gates() -> None:
    """Promotion must require parity, schemas, benchmarks, and fallback behavior."""
    strategy = _strategy_text()

    for gate in (
        "Reference parity",
        "Schema compatibility",
        "Benchmark evidence",
        "Fallback behavior",
        "Deterministic tests",
        "No ABI leakage",
        "first-call compilation time",
        "steady-state runtime",
        "Rust-native runtime",
    ):
        assert gate in strategy
