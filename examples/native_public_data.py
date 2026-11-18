"""Fit descriptive curves to a frozen, attributed public Internet-use series.

The World Bank/ITU series measures recent Internet use (% of population), not
cumulative first adoption. This offline example preserves missing calendar years,
fits only the training period, and compares later observations with a persistence
baseline. This retrospective split uses the November 2026 data vintage, not
as-of-2015 information. Numerical convergence is not scientific validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import polars as pl

from innovate import fit

DATASET = Path(__file__).resolve().parents[1] / "docs/datasets/wdi-aus-internet-20261116"


def load_observations(directory: Path = DATASET) -> tuple[pl.DataFrame, dict]:
    """Validate a frozen public-data extract and retain its missing observations.

    Args:
        directory: Directory containing the attributed manifest, raw API response
            and derived CSV. No files are downloaded or rewritten.

    Returns:
        The complete calendar-year frame and provenance manifest. ``observed`` is
        in percentage points; nulls remain missing and ``time`` is years since 1990.

    Raises:
        ValueError: If file hashes, source identity, transformation or shape differ.
        OSError: If a required local dataset file cannot be read.
        json.JSONDecodeError: If the manifest or API response is malformed.
    """
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["country_iso3"] != "AUS"
        or manifest["indicator"] != "IT.NET.USER.ZS"
        or manifest["unit"] != "percent_of_population"
        or manifest["time_unit"] != "year"
        or manifest["time_origin"] != 1990
        or manifest["license"] != "CC-BY-4.0"
        or set(manifest["files"]) != {"raw.json", "observations.csv"}
    ):
        raise ValueError("The lesson requires its declared country, indicator, units and licensed extract.")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen dataset checksum mismatch: {name}")
    metadata, records = json.loads((directory / "raw.json").read_text())
    if metadata["pages"] != 1 or metadata["total"] != len(records):
        raise ValueError("The frozen response must contain its complete requested page.")
    expected_rows = []
    for record in sorted(records, key=lambda row: int(row["date"])):
        if record["countryiso3code"] != "AUS" or record["indicator"]["id"] != "IT.NET.USER.ZS":
            raise ValueError("A source record has a different country or indicator.")
        year, value = int(record["date"]), record["value"]
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError("Internet use must be missing or a finite percentage from zero to 100.")
        expected_rows.append({"year": year, "time": float(year - 1990), "observed": value})
    with (directory / "observations.csv").open(newline="") as stream:
        actual_rows = [
            {
                "year": int(row["year"]),
                "time": float(row["time"]),
                "observed": float(row["observed"]) if row["observed"] else None,
            }
            for row in csv.DictReader(stream)
        ]
    if actual_rows != expected_rows or len({row["year"] for row in actual_rows}) != len(actual_rows):
        raise ValueError("Derived observations do not match the declared raw-data transformation.")
    missing = [row["year"] for row in actual_rows if row["observed"] is None]
    if (
        missing != manifest["missing_years"]
        or len(actual_rows) != manifest["raw_rows"]
        or len(actual_rows) - len(missing) != manifest["observed_rows"]
        or metadata["lastupdated"] != manifest["source_last_updated"]
    ):
        raise ValueError("The dataset inventory or source revision is stale.")
    return pl.DataFrame(actual_rows, schema={"year": pl.Int64, "time": pl.Float64, "observed": pl.Float64}), manifest


def run_comparison(cutoff_year: int = 2015) -> dict:
    """Run the same native computation from the CLI and reactive notebook.

    Args:
        cutoff_year: Last calendar year included in training, from 2010 to 2020.
            Later nonmissing observations are held out; no observations are shuffled.

    Returns:
        Fresh JSON-compatible provenance, scores, coefficients and holdout rows.
        Errors are in percentage points. Only native point estimates are returned.

    Raises:
        ValueError: If the cutoff or frozen observations violate this lesson's contract.
        innovate.NativeError: If native fitting or prediction fails.
        AssertionError: If convergence, time separation or prediction invariants fail.
    """
    if type(cutoff_year) is not int or not 2010 <= cutoff_year <= 2020:
        raise ValueError("Choose an integer training cutoff from 2010 to 2020.")
    complete, manifest = load_observations()
    observed = complete.filter(pl.col("observed").is_not_null())
    training = observed.filter(pl.col("year") <= cutoff_year)
    holdout = observed.filter(pl.col("year") > cutoff_year)
    assert training["year"].max() < holdout["year"].min()
    assert training.height >= 10 and holdout.height >= 4
    assert observed["time"].to_list() == [float(year - 1990) for year in observed["year"]]
    scores, parameters, predictions = [], {}, []
    target = holdout["observed"].to_list()
    for model in ("logistic", "gompertz", "persistence"):
        if model == "persistence":
            values = [training["observed"][-1]] * holdout.height
            converged = None
        else:
            fitted = fit(model, training.select("time", "observed"))
            converged = fitted.diagnostics["optimizer"]["converged"]
            assert converged is True, f"Inspect the {model} fit before interpreting its predictions."
            parameters[model] = fitted.parameters
            forecast = fitted.predict(holdout.select("year", "time"))
            assert forecast["year"].to_list() == holdout["year"].to_list()
            values = forecast["prediction"].to_list()
        assert len(values) == len(target) and all(math.isfinite(value) for value in values)
        residuals = [predicted - actual for predicted, actual in zip(values, target, strict=True)]
        scores.append(
            {
                "model": model,
                "holdout_mae": sum(map(abs, residuals)) / len(residuals),
                "holdout_rmse": math.sqrt(sum(value * value for value in residuals) / len(residuals)),
                "holdout_bias": sum(residuals) / len(residuals),
                "converged": converged,
            }
        )
        predictions.extend(
            {"model": model, "year": year, "observed": actual, "prediction": predicted}
            for year, actual, predicted in zip(holdout["year"], target, values, strict=True)
        )
    return {
        "dataset": manifest["id"],
        "raw_sha256": manifest["files"]["raw.json"],
        "cutoff_year": cutoff_year,
        "training_rows": training.height,
        "holdout_rows": holdout.height,
        "missing_years": manifest["missing_years"],
        "scores": scores,
        "parameters": parameters,
        "predictions": predictions,
    }


def main() -> None:
    """Check the default calendar holdout and print a stable tutorial result."""
    comparison = run_comparison()
    assert comparison["training_rows"] == 23 and comparison["holdout_rows"] == 9
    assert comparison["missing_years"] == [2002, 2003, 2004, 2025]
    score = {row["model"]: row["holdout_mae"] for row in comparison["scores"]}
    # A useful lesson can expose poor forecasting. Do not require the fitted
    # curves to beat a baseline when these observations show the opposite.
    assert score["persistence"] < score["gompertz"] < score["logistic"]
    print(
        json.dumps(
            {
                "cutoff_year": comparison["cutoff_year"],
                "training_rows": comparison["training_rows"],
                "holdout_rows": comparison["holdout_rows"],
                "missing_years": comparison["missing_years"],
                "holdout_mae": {model: round(value, 3) for model, value in score.items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
