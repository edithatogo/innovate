"""Tests for the benchmark model-card schema."""

from __future__ import annotations

import pytest

from innovate.benchmarks import (
    ModelCard,
    get_model_card,
    list_model_cards,
)
from innovate.capabilities import get_model_registry


def test_model_cards_cover_all_stable_model_capabilities() -> None:
    """Stable model capabilities should each have a synchronized model card."""
    registry = get_model_registry()
    stable_keys = {key for key, capability in registry.items() if capability.stability == "stable"}

    cards = list_model_cards()

    assert set(cards) == stable_keys
    assert all(isinstance(card, ModelCard) for card in cards.values())


def test_model_card_schema_is_complete() -> None:
    """A model card should capture the model, inputs, outputs, diagnostics, and limits."""
    card = get_model_card("bass")

    assert card.model_key == "bass"
    assert card.model_name == "BassModel"
    assert card.family == "diffusion"
    assert card.stability == "stable"
    assert card.assumptions
    assert card.inputs
    assert card.outputs
    assert card.diagnostics
    assert card.limitations
    assert card.benchmark_case_ids == ("bass_smoke_adoption",)
    assert card.validate() is None

    payload = card.to_dict()
    assert payload["model_key"] == "bass"
    assert payload["benchmark_case_ids"] == ["bass_smoke_adoption"]


def test_model_card_validation_rejects_incomplete_cards() -> None:
    """Missing required model-card fields should be rejected."""
    with pytest.raises(ValueError, match="assumptions"):
        ModelCard(
            model_key="bass",
            model_name="BassModel",
            family="diffusion",
            stability="stable",
            summary="Model card used for validation.",
            assumptions=(),
            inputs=("time",),
            outputs=("adoption",),
            diagnostics=("r2",),
            limitations=("Synthetic only.",),
            benchmark_case_ids=("bass_smoke_adoption",),
        )
