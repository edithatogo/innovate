"""Tests for the XLA backend strategy documentation."""

from __future__ import annotations

from pathlib import Path


def test_xla_backend_strategy_doc_exists_and_is_linked() -> None:
    """The XLA policy should be a first-class Sphinx document."""
    strategy = Path("docs/source/xla_backend_strategy.rst")
    index = Path("docs/source/index.rst").read_text()
    adr = Path("docs/source/adr.rst").read_text()
    principles = Path("docs/architecture_principles.md").read_text()
    roadmap = Path("docs/architecture_modernization_roadmap.md").read_text()

    assert strategy.is_file()
    assert "xla_backend_strategy" in index
    assert "xla_backend_strategy" in adr
    assert "XLA Backend Strategy" in principles
    assert "docs/source/xla_backend_strategy.rst" in roadmap


def test_xla_backend_strategy_defines_eligibility_and_rejection_rules() -> None:
    """The strategy should say when XLA is appropriate and when to reject it."""
    strategy = Path("docs/source/xla_backend_strategy.rst").read_text()

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
    strategy = Path("docs/source/xla_backend_strategy.rst").read_text()

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
    strategy = Path("docs/source/xla_backend_strategy.rst").read_text()

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
