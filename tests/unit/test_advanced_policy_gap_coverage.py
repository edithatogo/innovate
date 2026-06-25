"""Tests for advanced policy, competition, and substitution gap coverage.

These tests verify that capability registries, docs, model cards,
and schema status exist for each targeted model family.
They should fail initially (red phase) until gaps are filled.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

CAPABILITIES_SRC = ROOT / "src" / "innovate" / "capabilities.py"
GAP_INVENTORY = (
    ROOT / "conductor" / "tracks" / "advanced_policy_competition_substitution_20260625" / "gap_inventory.md"
)


def test_lock_in_model_registered_in_capabilities() -> None:
    """LockInModel should have a capability entry in _MODEL_REGISTRY."""
    text = CAPABILITIES_SRC.read_text(encoding="utf-8")
    # LockInModel should be registered
    assert "lock_in" in text or "LockInModel" in text, (
        "LockInModel is missing from capability registry"
    )


def test_path_dependence_module_exports() -> None:
    """innovate.path_dependence should export LockInModel."""
    init_file = ROOT / "src" / "innovate" / "path_dependence" / "__init__.py"
    text = init_file.read_text(encoding="utf-8")
    assert "LockInModel" in text, (
        "path_dependence/__init__.py does not export LockInModel"
    )


def test_equilibrium_method_on_lotka_volterra() -> None:
    """LotkaVolterraCompetition should have an equilibrium() method."""
    src_file = ROOT / "src" / "innovate" / "dynamics" / "competition" / "lotka_volterra.py"
    if not src_file.exists():
        src_file = ROOT / "src" / "innovate" / "compete" / "lotka_volterra.py"
    text = src_file.read_text(encoding="utf-8")
    assert "def equilibrium" in text, (
        "LotkaVolterraModel missing equilibrium() method"
    )


def test_equilibrium_method_on_multi_product() -> None:
    """MultiProductDiffusionModel should have an equilibrium() method."""
    mp_file = ROOT / "src" / "innovate" / "compete" / "multi_product.py"
    text = mp_file.read_text(encoding="utf-8")
    assert "def equilibrium" in text, (
        "MultiProductDiffusionModel missing equilibrium() method"
    )


def test_cross_elasticity_output_on_competition() -> None:
    """Competition models should expose cross-elasticity computation."""
    src_file = ROOT / "src" / "innovate" / "compete" / "competition.py"
    if src_file.exists():
        text = src_file.read_text(encoding="utf-8")
    else:
        src_file = ROOT / "src" / "innovate" / "dynamics" / "competition.py"
        text = src_file.read_text(encoding="utf-8")
    assert "cross_elasticity" in text, (
        "No cross-elasticity method found in competition module"
    )


def test_threshold_diagnostics_on_substitution() -> None:
    """Substitution models should expose threshold diagnostics."""
    fisher_pry = ROOT / "src" / "innovate" / "substitute" / "fisher_pry.py"
    norton_bass = ROOT / "src" / "innovate" / "substitute" / "norton_bass.py"
    if fisher_pry.exists():
        text = fisher_pry.read_text(encoding="utf-8")
        assert "threshold" in text.lower(), (
            "FisherPryModel missing threshold diagnostics"
        )
    if norton_bass.exists():
        text = norton_bass.read_text(encoding="utf-8")
        assert "threshold" in text.lower(), (
            "NortonBassModel missing threshold diagnostics"
        )


def test_network_diffusion_intervention_api() -> None:
    """NetworkDiffusionModel should support intervention nodes."""
    net_file = ROOT / "src" / "innovate" / "models" / "network.py"
    text = net_file.read_text(encoding="utf-8")
    assert "intervention" in text.lower() or "set_intervention" in text, (
        "NetworkDiffusionModel missing intervention node API"
    )


def test_gap_inventory_exists() -> None:
    """The gap inventory document should exist."""
    assert GAP_INVENTORY.exists(), (
        "Gap inventory document does not exist"
    )


def test_gap_inventory_covers_all_families() -> None:
    """Gap inventory should cover all required model families."""
    text = GAP_INVENTORY.read_text(encoding="utf-8")
    required_sections = [
        "Policy Diffusion",
        "Competition",
        "Substitution",
        "Network Diffusion",
        "Multi-Product",
        "Composite",
        "Advanced Runtime",
    ]
    for section in required_sections:
        assert section in text, (
            f"Gap inventory missing section: {section}"
        )


def test_starlight_docs_for_advanced_policy() -> None:
    """Starlight docs should exist for advanced policy modeling."""
    docs_dir = ROOT / "docs" / "astro-site" / "src" / "content" / "docs"
    hits: list[str] = []
    for f in docs_dir.rglob("*.md"):
        if "policy" in f.stem.lower() or "competition" in f.stem.lower() or "substitution" in f.stem.lower():
            hits.append(f.name)
    assert len(hits) >= 1, (
        "No Starlight docs found for advanced policy/competition/substitution modeling"
    )