"""E02 uncertainty-preserving Voiage decision-binding handoff tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def bind_decision(binding: dict[str, Any]) -> dict[str, Any]:
    """Validate DecisionBinding and compute exact two-strategy EVPI over correlated draws."""
    spec = json.loads(Path("specs/innovation-decision-binding-v1.json").read_text(encoding="utf-8"))
    for field in spec["required_fields"]:
        if field not in binding:
            raise ValueError(f"missing required field: {field}")
    if binding["fit_status"] != "valid":
        raise ValueError("cannot promote invalid fit to Voiage decision binding")
    if binding["units"] in spec["forbidden_raw_units"] or "net_benefit" not in str(binding["units"]):
        raise ValueError("prescription counts or unmapped units cannot become Voiage utilities")

    strategies = binding["strategies"]
    if len(strategies) < 2:
        raise ValueError("at least two strategies required")

    draw_ids = [d["draw_id"] for d in strategies[0]["draws"]]
    for strat in strategies[1:]:
        if [d["draw_id"] for d in strat["draws"]] != draw_ids:
            raise ValueError("correlated draw_ids must align across strategies")

    n_draws = len(draw_ids)
    mean_nb = [sum(d["net_benefit"] for d in s["draws"]) / n_draws for s in strategies]
    best_expected = max(mean_nb)
    expected_best = sum(max(s["draws"][i]["net_benefit"] for s in strategies) for i in range(n_draws)) / n_draws
    evpi = max(0.0, expected_best - best_expected)
    return {"evpi": evpi, "best_expected_nb": best_expected, "n_draws": n_draws}


def test_correlated_draws_evpi_and_rejection_of_raw_prescriptions():
    valid_binding = {
        "schema_version": "1.0",
        "perspective": "healthcare_payer",
        "population": 10000,
        "horizon": 10.0,
        "discount_rate": 0.035,
        "units": "net_benefit_usd",
        "fit_status": "valid",
        "strategies": [
            {"id": "status_quo", "draws": [{"draw_id": 0, "net_benefit": 10.0}, {"draw_id": 1, "net_benefit": 20.0}]},
            {"id": "new_therapy", "draws": [{"draw_id": 0, "net_benefit": 15.0}, {"draw_id": 1, "net_benefit": 12.0}]},
        ],
    }
    res = bind_decision(valid_binding)
    # E[NB_sq]=15, E[NB_new]=13.5 -> max=15; E[max(10,15), max(20,12)]=(15+20)/2=17.5 -> EVPI=2.5
    assert res["evpi"] == pytest.approx(2.5)

    bad_units = dict(valid_binding, units="prescriptions")
    with pytest.raises(ValueError, match="utilities"):
        bind_decision(bad_units)

    swapped = {
        **valid_binding,
        "strategies": [
            valid_binding["strategies"][0],
            {"id": "new_therapy", "draws": [{"draw_id": 1, "net_benefit": 12.0}, {"draw_id": 0, "net_benefit": 15.0}]},
        ],
    }
    with pytest.raises(ValueError, match="align"):
        bind_decision(swapped)

    with pytest.raises(ValueError, match="invalid fit"):
        bind_decision(dict(valid_binding, fit_status="invalid"))
