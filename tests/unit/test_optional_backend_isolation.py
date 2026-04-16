"""Tests for optional backend and dependency isolation."""

from __future__ import annotations

import importlib
import sys

import pytest

import innovate


def test_capability_registries_describe_backends_and_fitters():
    """Backend and fitter registries should expose the support policy."""
    backend_registry = innovate.get_backend_registry()
    fitter_registry = innovate.get_fitter_registry()

    assert backend_registry["numpy"].available is True
    assert backend_registry["jax"].stability == "experimental"
    assert "diffrax" in backend_registry["jax"].optional_dependencies

    assert fitter_registry["scipy"].supported_backends == ("numpy",)
    assert fitter_registry["batched"].supported_backends == ("numpy", "jax")
    assert fitter_registry["bayesian"].stability == "experimental"
    assert "blackjax" in fitter_registry["bayesian"].optional_dependencies

    assert innovate.get_model_capability("bass").supported_backends == ("numpy", "jax")

    with pytest.raises(KeyError, match="Unknown backend capability"):
        innovate.get_backend_capability("does_not_exist")

    with pytest.raises(KeyError, match="Unknown fitter capability"):
        innovate.get_fitter_capability("does_not_exist")


def test_missing_optional_fitters_raise_clear_import_errors(monkeypatch):
    """Optional fitters should fail with an explicit install hint when unavailable."""
    for module_name in (
        "innovate.fitters",
        "innovate.fitters.bayesian_fitter",
        "innovate.fitters.blackjax_fitter",
        "innovate.fitters.jax_fitter",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    for module_name in (
        "jax",
        "jaxlib",
        "jaxopt",
        "blackjax",
        "arviz",
        "diffrax",
    ):
        monkeypatch.setitem(sys.modules, module_name, None)

    fitters = importlib.import_module("innovate.fitters")

    with pytest.raises(ImportError, match="Install innovate\\[bayesian\\]"):
        fitters.BayesianFitter()

    with pytest.raises(ImportError, match="Install innovate\\[bayesian\\]"):
        fitters.BlackJaxFitter()

    with pytest.raises(ImportError, match="Install innovate\\[jax\\]"):
        fitters.JaxFitter()
