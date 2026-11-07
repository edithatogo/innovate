"""Rust-native diffusion modelling with Polars and Arrow IPC.

This breaking development API exposes only native capabilities. Unported
features remain migration obligations and never fall back to Python.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import polars as pl

from . import _native

__version__ = version("innovate")
__all__ = ["FitResult", "NativeError", "capabilities", "execute", "fit", "predict"]


class NativeError(ValueError):
    """Structured scientific or capability error returned by the Rust core."""

    def __init__(self, error: dict[str, Any]):
        self.code = str(error["code"])
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


def capabilities() -> pl.DataFrame:
    """Return currently implemented native models, with bounded support metadata."""
    return pl.DataFrame(execute("discover_models")["result"]["models"])


def _ipc(frame: pl.DataFrame | pl.LazyFrame) -> bytes:
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("expected a Polars DataFrame or LazyFrame")
    return frame.write_ipc(None).getvalue()


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
