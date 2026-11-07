"""Validate the historical diffusion fixture through the installed kernel."""

from pathlib import Path

import polars as pl
import pytest

import innovate

FIXTURE = Path(__file__).resolve().parents[2] / "specs/ecosystem/voiage/uncertainty/diffusion_v1"


def test_historical_draws_round_trip_and_reject_reordering():
    parameters = pl.read_csv(FIXTURE / "parameter_draws.csv")
    trajectories = pl.read_csv(FIXTURE / "adoption_trajectories.csv")
    strategies = []
    for (scenario,), rows in trajectories.group_by("scenario_id", maintain_order=True):
        draws = []
        for (draw_id,), draw in rows.group_by("draw_id", maintain_order=True):
            values = parameters.filter((pl.col("scenario_id") == scenario) & (pl.col("draw_id") == draw_id))
            draws.append(
                {
                    "draw_id": draw_id,
                    "time": draw["time"].to_list(),
                    "adoption": draw["adoption"].to_list(),
                    "parameters": values.select(pl.col("parameter_name").alias("name"), "unit", "value").to_dicts(),
                }
            )
        strategies.append(
            {
                "scenario_id": scenario,
                "time_unit": "period",
                "provenance": "voiage_diffusion_uncertainty_v1",
                "draws": draws,
            }
        )

    assert [len(strategy["draws"]) for strategy in strategies] == [3, 3]
    assert innovate.execute("validate_paired_outcomes", strategies=strategies)["result"] == {"strategies": strategies}
    exported = innovate.paired_outcomes({"strategies": strategies})
    keys = ["scenario_id", "draw_id", "time", "adoption"]
    assert exported.select(keys).rows() == trajectories.select(keys).rows()

    strategies[1]["draws"].reverse()
    with pytest.raises(ValueError, match="matching ordered draw IDs"):
        innovate.paired_outcomes({"strategies": strategies})
    with pytest.raises(innovate.NativeError, match="matching ordered draw IDs"):
        innovate.execute("validate_paired_outcomes", strategies=strategies)
