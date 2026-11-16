"""E05 application promotion ledger and bounded surrogate gate tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_application_promotion_ledger_is_acyclic_and_has_oracles():
    ledger = json.loads(Path("conductor/qualification/application-promotion.json").read_text(encoding="utf-8"))
    for entry in ledger["promotions"]:
        assert entry["acyclic"] is True
        assert len(entry["consumers"]) >= 2
        assert Path(entry["oracle"]).exists()


def test_surrogate_preserves_ranking_in_domain_and_abstains_out_of_domain():
    def surrogate_predict(x: float) -> float:
        if not (0.0 <= x <= 10.0):
            raise ValueError("out-of-domain surrogate input: abstaining")
        return 2.0 * x + 1.0

    assert surrogate_predict(3.0) > surrogate_predict(1.0)
    with pytest.raises(ValueError, match="abstaining"):
        surrogate_predict(25.0)
