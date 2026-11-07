"""Rust-native diffusion modelling with Polars and Arrow IPC.

This breaking development API exposes only native capabilities. Unported
features remain migration obligations and never fall back to Python.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from . import _native

__version__ = version("innovate")
__all__ = [
    "FitResult",
    "NativeError",
    "categorize_adopters",
    "aggregate_weekly",
    "aggregate_time_series",
    "cumulative_sum",
    "rolling_average",
    "seasonal_decomposition",
    "sarima_fit",
    "treatment_contrast_frame",
    "policy_metadata_frame",
    "ingest_polars",
    "ingest_local",
    "capabilities",
    "execute",
    "fit",
    "fit_batch",
    "fit_frames",
    "predict",
    "predict_batch",
    "predict_frames",
]


class NativeError(ValueError):
    """Structured scientific or capability error returned by the Rust core."""

    def __init__(self, error: dict[str, Any]):
        self.code = str(error["code"])
        self.validation: dict[str, Any] | None = error.get("validation")
        super().__init__(f"{self.code}: {error['message']}")


def _decode(response: str) -> dict[str, Any]:
    result = json.loads(response)
    if result.get("error") is not None:
        raise NativeError(result["error"])
    return result


def execute(operation: str, model: str | None = None, **payload: Any) -> dict[str, Any]:
    """Execute a versioned native request; advanced payloads may be unavailable."""
    request = {"schema_version": "2.0", "operation": operation, "model_key": model, "payload": payload}
    return _decode(_native.execute(json.dumps(request, allow_nan=False)))


def fit_batch(model: str, datasets: list[dict[str, Any]], *, workers: int = 1) -> list[dict[str, Any]]:
    """Fit independent native payloads and return ordered results or errors."""
    return execute("fit_batch", model, datasets=datasets, workers=workers)["result"]["datasets"]


def predict_batch(model: str, datasets: list[dict[str, Any]], *, workers: int = 1) -> list[dict[str, Any]]:
    """Predict on independent grids using the state in each payload."""
    return execute("predict_batch", model, datasets=datasets, workers=workers)["result"]["datasets"]


def capabilities() -> pl.DataFrame:
    """Return currently implemented native models, with bounded support metadata."""
    return pl.DataFrame(execute("discover_models")["result"]["models"])


def _ipc(frame: pl.DataFrame | pl.LazyFrame) -> bytes:
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("expected a Polars DataFrame or LazyFrame")
    return frame.write_ipc(None).getvalue()


def execute_scenario_grid(data: pl.DataFrame | pl.LazyFrame, **payload: Any) -> dict[str, Any]:
    """Execute scenarios on the shared time column of a Polars frame."""
    return _decode(_native.scenario_grid_ipc(_ipc(data), json.dumps(payload, allow_nan=False)))


def predict(
    model: str,
    data: pl.DataFrame | pl.LazyFrame,
    parameters: dict[str, float],
    *,
    model_options: dict[str, Any] | None = None,
    quantity: str = "level",
    covariate_rate_columns: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Append native predictions, preserving row order and identity columns.

    Requires a finite, strictly increasing numeric ``time`` column. IPC transport
    currently copies buffers; it is not claimed to be zero-copy. A LazyFrame is
    collected explicitly at the native execution boundary.
    """
    options = {"model_options": model_options if model_options is not None else {}, "quantity": quantity}
    if covariate_rate_columns is not None:
        options["covariate_rate_columns"] = covariate_rate_columns
    output = _native.predict_ipc(
        _ipc(data), model, json.dumps(parameters, allow_nan=False), json.dumps(options, allow_nan=False)
    )
    return pl.read_ipc(io.BytesIO(output))


@dataclass(frozen=True)
class FitResult:
    """Native fit envelope; point estimates do not imply calibrated uncertainty."""

    model: str
    envelope: dict[str, Any]

    @property
    def parameters(self) -> dict[str, float]:
        return dict(self.envelope["result"]["parameters"])

    @property
    def diagnostics(self) -> dict[str, Any]:
        return dict(self.envelope["result"]["diagnostics"])

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        quantity: str = "level",
        covariate_rate_columns: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        options = self.envelope["result"]["state"]["constructor_kwargs"]
        return predict(
            self.model,
            data,
            self.parameters,
            model_options=options,
            quantity=quantity,
            covariate_rate_columns=covariate_rate_columns,
        )

    def categorize_adopters(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        """Classify a single nonnegative native rate series on the supplied grid."""
        return categorize_adopters(self.predict(data, quantity="rate"), rate_column="rate")


def fit(
    model: str,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    model_options: dict[str, Any] | None = None,
    observed_columns: list[str] | None = None,
    fit_options: dict[str, Any] | None = None,
) -> FitResult:
    """Fit native parameters to finite ``time`` and ``observed`` columns."""
    options: dict[str, Any] = {
        "model_options": model_options if model_options is not None else {},
        "fit_options": fit_options if fit_options is not None else {},
    }
    if observed_columns is not None:
        options["observed_columns"] = observed_columns
    return FitResult(model, _decode(_native.fit_ipc(_ipc(data), model, json.dumps(options, allow_nan=False))))


def fit_frames(
    model: str,
    datasets: Iterable[pl.DataFrame | pl.LazyFrame],
    *,
    model_options: dict[str, Any] | None = None,
    observed_columns: list[str] | None = None,
    fit_options: dict[str, Any] | None = None,
) -> list[FitResult | Exception]:
    """Fit each frame independently and keep its result or input error in order."""
    results: list[FitResult | Exception] = []
    for frame in datasets:
        try:
            results.append(
                fit(
                    model,
                    frame,
                    model_options=model_options,
                    observed_columns=observed_columns,
                    fit_options=fit_options,
                )
            )
        except (ValueError, TypeError, pl.exceptions.PolarsError) as error:
            results.append(error)
    return results


def predict_frames(
    datasets: Iterable[tuple[FitResult | Exception, pl.DataFrame | pl.LazyFrame]],
    *,
    quantity: str = "level",
    covariate_rate_columns: dict[str, str] | None = None,
) -> list[pl.DataFrame | Exception]:
    """Forecast each frame with its paired fit, carrying failed fits forward."""
    results: list[pl.DataFrame | Exception] = []
    for fitted, frame in datasets:
        if isinstance(fitted, Exception):
            results.append(fitted)
            continue
        try:
            results.append(fitted.predict(frame, quantity=quantity, covariate_rate_columns=covariate_rate_columns))
        except (ValueError, TypeError, pl.exceptions.PolarsError) as error:
            results.append(error)
    return results


def categorize_adopters(data: pl.DataFrame | pl.LazyFrame, *, rate_column: str = "adoption_rate") -> pl.DataFrame:
    """Append historical adopter categories using supplied nonnegative rates."""
    output = _native.categorize_adopters_ipc(_ipc(data), rate_column)
    return pl.read_ipc(io.BytesIO(output))


def cumulative_sum(data: pl.Series) -> pl.Series:
    """Compute the cumulative sum in the supplied sample order."""
    return data.cum_sum()


def rolling_average(data: pl.Series, window: int) -> pl.Series:
    """Return trailing means after a full window of observations is available."""
    if window < 1:
        raise ValueError("window must be positive")
    return data.rolling_mean(window_size=window, min_samples=window)


def aggregate_weekly(
    data: pl.DataFrame | pl.LazyFrame, value_columns: list[str], *, time_column: str = "time"
) -> pl.DataFrame:
    """Sum numeric columns in fixed seven-day windows anchored at the first timestamp."""
    return aggregate_time_series(data, "W", value_columns, time_column=time_column)


def aggregate_time_series(
    data: pl.DataFrame | pl.LazyFrame, frequency: str, value_columns: list[str], *, time_column: str = "time"
) -> pl.DataFrame:
    """Sum daily, anchored weekly, or calendar month-end observations."""
    periods = {"D": "1d", "W": "168h", "M": "1mo", "ME": "1mo"}
    if frequency not in periods:
        raise ValueError("frequency must be D, W, M or ME")
    frame = data.collect() if isinstance(data, pl.LazyFrame) else data
    timestamps = frame[time_column]
    if not timestamps.dtype.is_temporal() or timestamps.dtype.base_type() not in (pl.Date, pl.Datetime):
        raise ValueError("aggregation requires Date or Datetime timestamps")
    if timestamps.null_count():
        raise ValueError("aggregation timestamps must not contain nulls")
    if not value_columns or len(set(value_columns)) != len(value_columns):
        raise ValueError("value_columns must contain distinct numeric columns")
    if any(not frame[name].dtype.is_numeric() for name in value_columns):
        raise ValueError("aggregation values must be numeric")
    every = periods[frequency]
    grouped = (
        frame.sort(time_column)
        .group_by_dynamic(
            time_column, every=every, start_by="datapoint" if frequency == "W" else "window", closed="left"
        )
        .agg(pl.col(value_columns).sum())
    )
    if frequency == "W" or grouped.is_empty():
        return grouped
    grouped = grouped.upsample(time_column, every=every).with_columns(pl.col(value_columns).fill_null(0))
    if frequency in ("M", "ME"):
        grouped = grouped.with_columns(pl.col(time_column).dt.offset_by("1mo").dt.offset_by("-1d"))
    return grouped


def seasonal_decomposition(
    data: pl.Series,
    *,
    period: int,
    trend_span: int | None = None,
    robust: bool = False,
    historical: bool = False,
    inner_iterations: int = 5,
    outer_iterations: int | None = None,
    seasonal_span: int | None = None,
    low_pass_span: int | None = None,
    seasonal_degree: int = 1,
    seasonal_jump: int = 1,
    trend_degree: int = 1,
    trend_jump: int = 1,
    low_pass_degree: int = 1,
    low_pass_jump: int = 1,
) -> pl.DataFrame:
    """Decompose evenly spaced samples with the native seasonal smoother.

    Set historical=True to use historical STL local fit and reweighting rules.
    In this mode, omitting trend_span derives it from period and seasonal_span.
    """
    result = execute(
        "preprocess_series",
        method="seasonal_decomposition",
        values=data.to_list(),
        period=period,
        window=trend_span,
        robust=robust,
        historical=historical,
        inner_iterations=inner_iterations,
        outer_iterations=(5 if robust else 0) if outer_iterations is None else outer_iterations,
        seasonal_span=seasonal_span,
        low_pass_span=low_pass_span,
        seasonal_degree=seasonal_degree,
        seasonal_jump=seasonal_jump,
        trend_degree=trend_degree,
        trend_jump=trend_jump,
        low_pass_degree=low_pass_degree,
        low_pass_jump=low_pass_jump,
    )["result"]
    return pl.DataFrame(result)


def sarima_fit(
    data: pl.Series,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    historical: bool = False,
) -> pl.Series:
    """Estimate a no-intercept SARIMA model and return aligned fitted values.

    Set historical=True to include the diffuse initialization predictions.
    """
    result = execute(
        "preprocess_series",
        method="sarima_fit",
        values=data.to_list(),
        order=order,
        seasonal_order=seasonal_order,
        historical=historical,
    )["result"]
    return pl.Series(data.name, result["fitted"], dtype=pl.Float64)


def treatment_contrast_frame(result: dict[str, Any]) -> pl.DataFrame:
    """Project a native treatment contrast into the historical Arrow result schema."""
    interval = result["bootstrap"]
    return pl.DataFrame(
        {
            "estimate": [result["overall"]["estimate"]],
            "ci_lower": [interval["ci_lower"] if interval is not None else None],
            "ci_upper": [interval["ci_upper"] if interval is not None else None],
            "n_bootstrap": [interval["n_bootstrap"] if interval is not None else 0],
        },
        schema={
            "estimate": pl.Float64,
            "ci_lower": pl.Float64,
            "ci_upper": pl.Float64,
            "n_bootstrap": pl.Int32,
        },
    )


def policy_metadata_frame(result: dict[str, Any]) -> pl.DataFrame:
    """Project native policy evidence into the historical model metadata schema."""
    return pl.DataFrame(
        [result["metadata"]],
        schema={
            "intervention_name": pl.String,
            "model_name": pl.String,
            "treatment_variable": pl.String,
            "outcome_variable": pl.String,
            "n_confounders": pl.Int32,
        },
    )


def ingest_polars(
    frame: pl.DataFrame | pl.LazyFrame,
    kind: str,
    *,
    provenance: dict[str, Any],
    unit: str | None = None,
    directed: bool = False,
) -> dict[str, Any]:
    """Convert a Polars table into a validated, checksummed dataset artifact."""
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    dataset: dict[str, Any] = {"kind": kind, "schema_version": "1.0", "provenance": provenance}
    if unit is not None:
        dataset["unit"] = unit

    def column(name: str, dtype: type[pl.DataType]) -> list[Any]:
        if name not in frame.columns:
            raise ValueError(f"missing required column: {name}")
        return frame[name].cast(dtype).to_list()

    if kind == "adoption":
        dataset.update(time=column("time", pl.Float64), adoption=column("adoption", pl.Float64))
        if "denominator" in frame.columns:
            dataset["denominator"] = column("denominator", pl.Float64)
    elif kind == "substitution":
        shares = [name for name in frame.columns if name.startswith("share")]
        if not shares:
            raise ValueError("substitution requires share columns")
        dataset.update(
            time=column("time", pl.Float64),
            share=frame.select(pl.col(shares).cast(pl.Float64)).rows(),
            product_labels=[
                name.removeprefix("share").lstrip("_") or f"product_{index}"
                for index, name in enumerate(shares)
            ],
        )
    elif kind == "competition":
        dataset.update(
            time=column("time", pl.Float64),
            value=column("value", pl.Float64),
            unit_id=column("unit_id", pl.String),
            product_id=column("product_id", pl.String),
        )
    elif kind == "policy_timing":
        dataset.update(
            event_times=column("event_times", pl.Float64),
            event_effects=column("event_effects", pl.Float64),
            event_labels=(
                column("event_labels", pl.String)
                if "event_labels" in frame.columns
                else [f"event_{i}" for i in range(frame.height)]
            ),
        )
    elif kind == "network_edges":
        dataset.update(
            source=column("source", pl.String),
            target=column("target", pl.String),
            weight=column("weight", pl.Float64) if "weight" in frame.columns else [1.0] * frame.height,
            directed=directed,
        )
    else:
        raise ValueError(f"unsupported dataset kind: {kind}")
    return execute("ingest_dataset", dataset=dataset)["result"]


def ingest_local(
    path: str | Path,
    kind: str,
    *,
    provenance: dict[str, Any],
    format: str | None = None,
    unit: str | None = None,
    directed: bool = False,
) -> dict[str, Any]:
    """Read a local CSV, Parquet or Arrow IPC file and validate it in Rust."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    file_format = (format or path.suffix).lstrip(".").lower()
    if file_format in {"csv", "txt"}:
        frame = pl.read_csv(path)
    elif file_format in {"parquet", "pq"}:
        frame = pl.read_parquet(path)
    elif file_format in {"arrow", "feather", "ipc"}:
        frame = pl.read_ipc(path)
    else:
        raise ValueError(f"unsupported local format: {file_format}")
    file_provenance = {
        **provenance,
        "transform_steps": [*provenance.get("transform_steps", []), f"load:{path.name}"],
        "extra": {**provenance.get("extra", {}), "path": str(path), "format": file_format},
    }
    return ingest_polars(frame, kind, provenance=file_provenance, unit=unit, directed=directed)
