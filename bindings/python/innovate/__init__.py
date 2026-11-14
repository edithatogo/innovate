"""Fit, predict and exchange diffusion data through the native Rust kernel.

The development API accepts Polars tables at tabular boundaries and JSON-compatible
objects at the versioned request boundary. Native model names and supported
features are discoverable with :func:`capabilities`. Unported features raise
errors; there is no fallback to the historical Python scientific implementations.

LazyFrames are collected when a tabular operation executes. Arrow IPC serialization
and deserialization copy buffers and do not promise zero-copy interchange. Numeric
``time`` values and model parameters must use mutually consistent units chosen by
the caller; the general fit and prediction wrappers do not infer or convert units.

Native request errors become :class:`NativeError`, while the tabular extension
raises ``ValueError`` for its transport, schema and prediction failures. Polars,
JSON serialization and local I/O errors can also propagate. A returned fit is
computational evidence, not a claim that its uncertainty or scientific assumptions
have been validated for a particular use.

Attributes:
    __version__: Installed Python distribution version.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from . import _native

__version__ = version("innovate")
__all__ = [
    "staggered_att_polars",
    "execute_scenario_grid",
    "export_ipc_file",
    "export_ipc_stream",
    "scenario_executions",
    "scenario_comparisons",
    "paired_outcomes",
    "FitResult",
    "NativeError",
    "categorize_adopters",
    "aggregate_weekly",
    "aggregate_time_series",
    "cumulative_sum",
    "rolling_average",
    "seasonal_decomposition",
    "sarima_fit",
    "bootstrap_forecast_frame",
    "treatment_contrast_frame",
    "policy_metadata_frame",
    "ingest_polars",
    "ingest_local",
    "fetch_oecd_dataset",
    "capabilities",
    "execute",
    "invoke_registered_extension",
    "fit",
    "fit_batch",
    "fit_frames",
    "predict",
    "predict_batch",
    "predict_frames",
]


class NativeError(ValueError):
    """Expose a native request failure as a Python ``ValueError`` subtype.

    Attributes:
        code: Machine-readable error code copied from the native error envelope.
        validation: Structured validation details when supplied by the kernel,
            otherwise ``None``. The nested object is not deep-copied.
    """

    def __init__(self, error: dict[str, Any]):
        """Construct an exception from the kernel's error object.

        Args:
            error: Mapping containing ``code`` and ``message`` and, optionally,
                ``validation`` details. Its code is converted to a string.

        Raises:
            KeyError: If the native error mapping lacks ``code`` or ``message``.
        """
        self.code = str(error["code"])
        self.validation: dict[str, Any] | None = error.get("validation")
        super().__init__(f"{self.code}: {error['message']}")


def _decode(response: str) -> dict[str, Any]:
    """Decode a native JSON envelope and raise its structured error, if present.

    Successful responses retain the complete envelope. JSON decoding errors and
    malformed envelope structure propagate rather than being replaced by defaults.
    """
    result = json.loads(response)
    if result.get("error") is not None:
        raise NativeError(result["error"])
    return result


def execute(operation: str, model: str | None = None, **payload: Any) -> dict[str, Any]:
    """Execute a schema-2.0 request and retain the complete native response.

    Use this low-level interface for operations that do not have a table wrapper.
    Payload fields are operation-specific and must obey the native contract;
    unsupported operations and model features fail rather than falling back.

    Args:
        operation: Native operation name, such as ``discover_models`` or
            ``predict_model``.
        model: Native model key, such as ``bass`` or ``logistic``. Use the default
            ``None`` for operations that do not select a model.
        **payload: JSON-compatible keyword fields placed in the request's
            ``payload`` object. All numeric values must be finite.

    Returns:
        Decoded response envelope, including ``result`` and native metadata.
        Successful envelopes may omit ``error``. A non-null native error is
        raised before this value is returned.

    Raises:
        NativeError: If the kernel rejects the request or cannot execute it.
        TypeError: If a payload value cannot be serialized as JSON.
        ValueError: If serialization encounters NaN, infinity or a circular object.

    Examples:
        >>> import innovate
        >>> response = innovate.execute("discover_models")
        >>> assert response.get("error") is None
        >>> assert "bass" in {model["key"] for model in response["result"]["models"]}
    """
    return _execute_payload(operation, model, payload)


def _execute_payload(operation: str, model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Serialize one explicit payload under schema 2.0 and decode the response.

    Keeping the payload mapping separate permits native fields that would clash
    with the public wrapper's Python parameter names. Non-finite JSON values are
    rejected before crossing the native boundary.
    """
    request = {"schema_version": "2.0", "operation": operation, "model_key": model, "payload": payload}
    return _decode(_native.execute(json.dumps(request, allow_nan=False)))


def invoke_registered_extension(
    callback: Callable[[dict[str, Any], Any], Any],
    selection: dict[str, Any],
    payload: Any,
) -> dict[str, Any]:
    """Resolve an extension, invoke its trusted callback, and validate the output.

    The callback runs Python code in the current process without sandboxing.
    Manifest resolution happens before invocation; native ingestion, alias,
    diagnostic or artifact validation happens afterward. Validation does not
    establish the safety or scientific validity of arbitrary callback behaviour.

    Args:
        callback: Trusted callable receiving the resolved manifest and ``payload``.
            Its return value must satisfy the selected extension point's contract.
        selection: JSON-compatible selection passed to ``resolve_extension_manifest``.
        payload: Application object passed unchanged to ``callback``; it is not
            serialized by this wrapper before callback invocation.

    Returns:
        Native validated ``result`` for the callback output, without the outer
        request envelope.

    Raises:
        NativeError: If manifest resolution or output validation fails.
        TypeError: If selection or callback output is not JSON-compatible.
        ValueError: If serialized fields contain non-finite or circular values.
        Exception: Any exception raised by the trusted callback propagates unchanged.
    """
    resolved = _execute_payload("resolve_extension_manifest", None, selection)["result"]
    output = callback(resolved, payload)
    if resolved["extension_point"] == "dataset_provider":
        return execute("ingest_dataset", dataset=output)["result"]
    if resolved["extension_point"] == "model_registry":
        return execute("resolve_extension_model_aliases", aliases=output)["result"]
    if resolved["extension_point"] == "diagnostics":
        return execute("summarize_extension_diagnostics", checks=output)["result"]
    return execute("import_extension_artifact", artifact=output)["result"]


def fit_batch(model: str, datasets: list[dict[str, Any]], *, workers: int = 1) -> list[dict[str, Any]]:
    """Fit independent native request payloads while retaining per-item failures.

    Args:
        model: Native model key shared by the batch.
        datasets: Ordered ``fit_model`` payload mappings, each containing its own
            ``inputs`` and any constructor or fitting options. These are native
            JSON payloads, not Polars frames; use ``fit_frames`` for tables.
        workers: Native worker count from 1 through 32, defaulting to 1. More
            workers do not change the returned dataset order.

    Returns:
        One mapping per input, in order, with ``result`` and ``error`` keys.
        Individual fitting failures are represented in these rows, not raised.

    Raises:
        NativeError: If the batch request itself is invalid, including the worker
            count or model-key shape.
        TypeError: If the request cannot be JSON-serialized.
        ValueError: If serialization encounters non-finite or circular values.
    """
    return execute("fit_batch", model, datasets=datasets, workers=workers)["result"]["datasets"]


def predict_batch(model: str, datasets: list[dict[str, Any]], *, workers: int = 1) -> list[dict[str, Any]]:
    """Predict independent native payloads with their own parameters or state.

    Args:
        model: Native model key shared by the batch.
        datasets: Ordered ``predict_model`` payload mappings with their own input
            grids and model parameters or fitted state. Use ``predict_frames``
            for pairs of ``FitResult`` objects and Polars tables.
        workers: Native worker count from 1 through 32; defaults to serial execution.

    Returns:
        Ordered mappings containing ``result`` and ``error`` for each dataset.
        An individual prediction failure is retained as its row's error.

    Raises:
        NativeError: If the outer batch request is invalid.
        TypeError: If a payload is not JSON-compatible.
        ValueError: If serialization encounters non-finite or circular values.
    """
    return execute("predict_batch", model, datasets=datasets, workers=workers)["result"]["datasets"]


def capabilities() -> pl.DataFrame:
    """Inspect model capabilities advertised by the installed native kernel.

    Returns:
        Polars table with one row per discovered model, including ``key`` and
        support metadata. Flags identify implemented operations; they do not
        establish suitability or calibration for a particular dataset.

    Raises:
        NativeError: If native model discovery fails.
        polars.exceptions.PolarsError: If discovery rows cannot form a table.
    """
    return pl.DataFrame(execute("discover_models")["result"]["models"])


def _ipc(frame: pl.DataFrame | pl.LazyFrame) -> bytes:
    """Collect a LazyFrame if needed, then serialize a Polars table to IPC bytes.

    Other input types raise ``TypeError``. Collection and IPC failures propagate
    from Polars; serialization materializes a buffer rather than sharing memory.
    """
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("expected a Polars DataFrame or LazyFrame")
    return frame.write_ipc(None).getvalue()


def export_ipc_file(data: pl.DataFrame | pl.LazyFrame | bytes) -> bytes:
    """Re-encode a table or Arrow IPC input as interoperable IPC file bytes.

    The native writer preserves compatible field annotations and schema metadata
    from IPC input. Materialized Polars tables can only supply metadata still
    represented in that table. This conversion copies and re-encodes buffers.

    Args:
        data: Polars DataFrame, LazyFrame, or complete IPC file/stream bytes.
            Lazy input is collected before serialization.

    Returns:
        IPC file bytes containing the table schema and record batches.

    Raises:
        TypeError: If ``data`` is neither bytes nor a supported Polars frame.
        ValueError: If native IPC reading or writing fails.
        polars.exceptions.PolarsError: If collection or initial serialization fails.
    """
    return _native.export_ipc_file(data if isinstance(data, bytes) else _ipc(data))


def export_ipc_stream(data: pl.DataFrame | pl.LazyFrame | bytes) -> bytes:
    """Re-encode a table or Arrow IPC input as an IPC stream with schema metadata.

    Args:
        data: Polars DataFrame, LazyFrame, or complete IPC file/stream bytes.
            Lazy input is collected, and IPC conversion copies buffers.

    Returns:
        IPC stream bytes carrying the schema and record batches. Existing IPC
        schema metadata and compatible field annotations are retained by the
        native writer; metadata already lost in a Polars table is not reconstructed.

    Raises:
        TypeError: If ``data`` is neither bytes nor a supported Polars frame.
        ValueError: If native IPC reading or writing fails.
        polars.exceptions.PolarsError: If collection or initial serialization fails.
    """
    return _native.export_ipc_stream(data if isinstance(data, bytes) else _ipc(data))


def fetch_oecd_dataset(
    dataset_id: str, dimension_key: str, *, start_date: str = "1960", end_date: str = "2020"
) -> pl.DataFrame:
    """Download OECD SDMX observations and parse the response with Polars.

    This operation performs a network request through the Rust client. The client
    uses a 30-second timeout, rejects redirects and limits the response to 16 MiB.
    Returned observations are not an immutable snapshot or an ingestion artifact.

    Args:
        dataset_id: OECD ``agency,dataflow[,version]`` reference. Dimension order
            must match the selected dataflow.
        dimension_key: Ordered SDMX dimensions separated by dots; empty segments
            select all codes and plus signs combine codes within a dimension.
        start_date: SDMX start-period code; defaults to ``"1960"``.
        end_date: SDMX end-period code; defaults to ``"2020"``.

    Returns:
        Parsed observation table. ``TIME_PERIOD`` is kept as a string and
        ``OBS_VALUE`` is parsed as Float64; other columns use Polars inference.

    Raises:
        NativeError: If request fields are invalid, the request fails or the
            upstream response violates the status, size or encoding requirements.
        polars.exceptions.PolarsError: If the response cannot be parsed as CSV.
    """
    result = execute(
        "fetch_oecd_dataset",
        dataset_id=dataset_id,
        dimension_key=dimension_key,
        start_date=start_date,
        end_date=end_date,
    )["result"]
    return pl.read_csv(
        io.BytesIO(result["csv"].encode("utf-8")),
        schema_overrides={"TIME_PERIOD": pl.String, "OBS_VALUE": pl.Float64},
    )


def execute_scenario_grid(data: pl.DataFrame | pl.LazyFrame, **payload: Any) -> dict[str, Any]:
    """Execute a native scenario grid using one shared table-derived time grid.

    Args:
        data: Polars table with a nonempty, finite, strictly increasing numeric
            ``time`` column without nulls. Lazy input is collected and IPC copies
            buffers. Other table columns are not injected into scenario payloads.
        **payload: Native scenario-grid fields, including a ``scenarios`` list.
            Each scenario needs a ``simulation`` mapping. Its ``inputs`` mapping
            must omit ``time`` and ``t`` because this wrapper supplies the grid.

    Returns:
        Complete native grid envelope with execution and comparison results.
        Inspect individual execution errors even when the outer request succeeds.

    Raises:
        NativeError: If the native grid request returns an outer error.
        ValueError: If the grid, scenario schema, IPC or JSON values are invalid.
        TypeError: If input is not a Polars frame or payload serialization fails.
        polars.exceptions.PolarsError: If collection or serialization fails.
    """
    return _decode(_native.scenario_grid_ipc(_ipc(data), json.dumps(payload, allow_nan=False)))


def scenario_executions(response: dict[str, Any]) -> pl.DataFrame:
    """Tabulate execution entries from a successful scenario-grid envelope.

    Args:
        response: Complete response from ``execute_scenario_grid``, with a null
            outer error and an ``executions`` list inside ``result``.

    Returns:
        Ordered rows with ``scenario_id``, ``model_key``, ``successful`` and
        ``execution_json``. The JSON column retains nested results and errors.
        An execution is successful only when it has no error and is not paused.

    Raises:
        ValueError: If the envelope or execution rows are malformed, or native IPC
            conversion or finite JSON serialization fails.
        TypeError: If response values are not JSON-compatible.
        polars.exceptions.PolarsError: If the resulting IPC cannot be read.
    """
    return pl.read_ipc(io.BytesIO(_native.scenario_executions_ipc(json.dumps(response, allow_nan=False))))


def scenario_comparisons(response: dict[str, Any]) -> pl.DataFrame:
    """Tabulate pair and ensemble comparisons from a scenario-grid envelope.

    Args:
        response: Complete successful grid envelope containing ``comparisons``
            under ``result``. Individual comparison errors may still be present.

    Returns:
        Rows in input order with zero-based ``comparison_index``, ``kind``
        (``pair`` or ``ensemble``), ``successful`` and ``comparison_json``. Nested
        outcomes and errors remain in the JSON column.

    Raises:
        ValueError: If the envelope or comparison shape is invalid, native IPC
            conversion fails, or JSON values are non-finite or circular.
        TypeError: If response values are not JSON-compatible.
        polars.exceptions.PolarsError: If the produced IPC cannot be read.
    """
    return pl.read_ipc(io.BytesIO(_native.scenario_comparisons_ipc(json.dumps(response, allow_nan=False))))


def paired_outcomes(payload: dict[str, Any]) -> pl.DataFrame:
    """Validate paired Bass trajectories and flatten them to adoption rows.

    Args:
        payload: Native paired-outcomes mapping with ``strategies``. Each strategy
            supplies its scenario ID, time unit, provenance and ordered draws;
            draws carry IDs, named parameter records, time and adoption arrays.
            Strategies must satisfy the native pairing and trajectory contract.

    Returns:
        Rows in strategy, draw and time order with ``scenario_id``, ``draw_id``,
        ``time_unit``, ``perspective``, ``provenance``, ``parameters_json``,
        ``time`` and ``adoption``. Parameter names, units and order are preserved
        in JSON; this conversion does not estimate new trajectories or uncertainty.

    Raises:
        ValueError: If pairing, model consistency, numeric values or IPC conversion
            fail validation, or JSON contains non-finite or circular values.
        TypeError: If payload values cannot be serialized as JSON.
        polars.exceptions.PolarsError: If the output IPC cannot be read.
    """
    return pl.read_ipc(io.BytesIO(_native.paired_outcomes_ipc(json.dumps(payload, allow_nan=False))))


def predict(
    model: str,
    data: pl.DataFrame | pl.LazyFrame,
    parameters: dict[str, float],
    *,
    model_options: dict[str, Any] | None = None,
    quantity: str = "level",
    covariate_rate_columns: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Append native predictions without changing the input row order.

    Input columns are retained in the returned table. A LazyFrame is collected at
    execution and IPC transport copies buffers. Numeric time and parameters must
    use consistent caller-selected units. A rate can be signed for models with
    declining levels; it is not automatically a measure of gross incidence.

    Args:
        model: Native model key, such as ``bass`` or ``logistic``.
        data: Polars table containing a nonempty, finite, strictly increasing
            numeric ``time`` column without nulls, plus any declared covariates.
        parameters: Finite parameter values keyed by the selected model's native
            parameter names. Parameters are not inferred or estimated here.
        model_options: Constructor options for the selected model; ``None`` means
            an empty mapping. Declared ``covariates`` name columns in ``data``.
        quantity: ``"level"`` (default) for levels or ``"rate"`` for native
            derivatives. The selected model must supply the requested quantity.
        covariate_rate_columns: Optional mapping from every declared Logistic
            control name to its numeric derivative column in ``data``. Supported
            only for Logistic rate predictions with declared covariates.

    Returns:
        A new DataFrame with the original columns and predictions. Scalar output
        adds ``prediction`` or ``rate``; multi-series output uses numbered names
        such as ``prediction_1`` or ``rate_1``. Output width follows the model.

    Raises:
        ValueError: If native validation or prediction fails, required columns are
            missing or invalid, output names collide, or JSON values are non-finite.
        TypeError: If data is not a Polars frame or options are not JSON-compatible.
        polars.exceptions.PolarsError: If lazy collection or IPC processing fails.

    Examples:
        >>> import innovate
        >>> import polars as pl
        >>> grid = pl.DataFrame({"time": [0.0, 1.0, 2.0], "id": ["a", "b", "c"]})
        >>> result = innovate.predict("logistic", grid, {"L": 100.0, "k": 1.0, "x0": 1.0})
        >>> assert result["id"].to_list() == ["a", "b", "c"]
        >>> assert result["prediction"][1] == 50.0
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
    """Keep the native fit envelope and expose parameters and reusable state.

    Instances prevent attribute reassignment, but the supplied envelope remains a
    mutable nested dictionary. Parameter and diagnostic accessors return shallow
    copies. Constructor options in the envelope are reused by ``predict``.
    Point estimates and convergence diagnostics do not establish calibrated
    uncertainty, identifiability or suitability for a scientific application.

    Attributes:
        model: Native model key used to generate predictions from this fit.
        envelope: Complete successful native fitting response, including the
            result's parameters, diagnostics and stored constructor state.
    """

    model: str
    envelope: dict[str, Any]

    @property
    def parameters(self) -> dict[str, float]:
        """Read fitted parameter estimates without exposing the original mapping.

        Returns:
            Shallow copy of the native result's parameter-name/value mapping.

        Raises:
            KeyError: If a manually constructed or mutated envelope lacks the
                result or parameters mapping.
            TypeError: If that mapping cannot be converted to a dictionary.
        """
        return dict(self.envelope["result"]["parameters"])

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Read the diagnostic fields supplied by the native fitting operation.

        Returns:
            Shallow copy of diagnostics such as objective or convergence evidence;
            available fields depend on the model and fitting operation.

        Raises:
            KeyError: If the envelope lacks a result or diagnostics mapping.
            TypeError: If the diagnostics cannot be converted to a dictionary.
        """
        return dict(self.envelope["result"]["diagnostics"])

    @property
    def event_profile(self) -> dict[str, Any] | None:
        """Read candidate scores and convergence for a profiled event fit.

        Returns:
            Shallow copy of ``event_profile``, or ``None`` when the native result
            contains no such profile. Scores are not calibrated confidence bounds.

        Raises:
            KeyError: If the envelope lacks its result mapping.
            TypeError: If a present profile cannot be converted to a dictionary.
        """
        profile = self.envelope["result"].get("event_profile")
        return dict(profile) if profile is not None else None

    @property
    def launch_profile(self) -> dict[str, Any] | None:
        """Read candidate scores for a profiled generation-launch fit.

        Returns:
            Shallow copy of ``launch_profile`` or ``None`` when it is absent.
            The profile retains the native candidate evaluation structure.

        Raises:
            KeyError: If the envelope lacks its result mapping.
            TypeError: If a present profile cannot be converted to a dictionary.
        """
        profile = self.envelope["result"].get("launch_profile")
        return dict(profile) if profile is not None else None

    @property
    def joint_time_profile(self) -> dict[str, Any] | None:
        """Read native candidate scores for a joint launch-and-event fit.

        Returns:
            Shallow copy of ``joint_time_profile`` or ``None`` when the fit did
            not return one. The accessor does not recompute the candidate search.

        Raises:
            KeyError: If the envelope lacks its result mapping.
            TypeError: If a present profile cannot be converted to a dictionary.
        """
        profile = self.envelope["result"].get("joint_time_profile")
        return dict(profile) if profile is not None else None

    def predict(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        *,
        quantity: str = "level",
        covariate_rate_columns: dict[str, str] | None = None,
    ) -> pl.DataFrame:
        """Predict with fitted parameters and the stored constructor options.

        Args:
            data: Polars frame with a nonempty, finite, strictly increasing
                numeric ``time`` column and all covariates required by this fit.
                Lazy input is collected and transported through copying IPC.
            quantity: ``"level"`` by default, or ``"rate"`` when supported by
                the model. Signed rates should not be interpreted as gross incidence.
            covariate_rate_columns: Optional map from declared Logistic controls
                to derivative columns, used only for covariate-aware rate output.

        Returns:
            Input columns and native prediction columns in the original row order,
            using the naming and shape contract of the module-level ``predict``.

        Raises:
            KeyError: If the stored envelope lacks parameters or constructor state.
            ValueError: If the native prediction, grid, options or output names
                violate the prediction contract.
            TypeError: If data or serializable options have unsupported types.
            polars.exceptions.PolarsError: If collection or IPC conversion fails.
        """
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
        """Predict a scalar rate series and assign historical adopter categories.

        Args:
            data: Polars time grid satisfying this fit's prediction contract.
                The model must produce one finite, nonnegative ``rate`` series.

        Returns:
            Input columns, predicted ``rate`` and a ``category`` string column.
            Categories depend on the supplied grid's sample-weighted rate mass,
            not individual-level adoption histories.

        Raises:
            ValueError: If scalar rates are unavailable, negative or have zero
                total mass, or input/output columns violate the native contract.
            KeyError: If the fit envelope lacks required prediction state.
            TypeError: If data or stored options cannot cross the native boundary.
            polars.exceptions.PolarsError: If table processing or IPC fails.
        """
        return categorize_adopters(self.predict(data, quantity="rate"), rate_column="rate")


def fit(
    model: str,
    data: pl.DataFrame | pl.LazyFrame,
    *,
    model_options: dict[str, Any] | None = None,
    observed_columns: list[str] | None = None,
    fit_options: dict[str, Any] | None = None,
) -> FitResult:
    """Estimate native model parameters from an ordered observation table.

    A LazyFrame is collected when fitting begins. The frame is serialized to
    Arrow IPC, so this interface copies buffers. Numeric time, observations and
    parameters must use consistent caller-specified units. Inspect diagnostics
    and model assumptions before interpreting a returned fit.

    Args:
        model: Native model key supported for fitting.
        data: Polars frame with a nonempty, finite, strictly increasing numeric
            ``time`` column, finite numeric observation columns without nulls, and
            any covariates declared in ``model_options``.
        model_options: Native constructor options; ``None`` becomes an empty
            mapping. These options are retained in the returned fitted state.
        observed_columns: Distinct observation column names in model output order.
            ``None`` uses ``["observed"]``; an empty list is invalid. One column
            becomes a vector and multiple columns become time-by-series rows.
        fit_options: Native fitting controls for the chosen model, such as
            supported initial parameters or iteration limits. ``None`` supplies
            an empty mapping and uses the kernel's model-specific defaults.

    Returns:
        FitResult containing the complete native fit envelope, parameter estimates,
        diagnostics and state for later prediction. A fit does not by itself
        certify calibrated uncertainty or adequate model specification.

    Raises:
        NativeError: If scientific fitting fails or a feature is unsupported.
        ValueError: If tabular inputs, options or finite JSON values are invalid.
        TypeError: If data is not a supported frame or options cannot be serialized.
        polars.exceptions.PolarsError: If collection or IPC serialization fails.

    Examples:
        >>> import math
        >>> import innovate
        >>> import polars as pl
        >>> time = [float(t) for t in range(20)]
        >>> observed = [100.0 / (1.0 + math.exp(-0.6 * (t - 8))) for t in time]
        >>> fitted = innovate.fit("logistic", pl.DataFrame({"time": time, "observed": observed}))
        >>> predictions = fitted.predict(pl.DataFrame({"time": time}))
        >>> assert max(abs(a - b) for a, b in zip(predictions["prediction"], observed)) < 0.01
    """
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
    """Fit an iterable of Polars frames sequentially and retain input failures.

    Each frame uses the same options, but is fitted independently. This wrapper
    does not enable the native batch worker pool; use ``fit_batch`` for native
    payload batches with worker control.

    Args:
        model: Native model key used for every frame.
        datasets: Iterable of DataFrames or LazyFrames; lazy inputs are collected
            individually at their fit boundary.
        model_options: Shared native constructor options; defaults to an empty mapping.
        observed_columns: Shared ordered observation column names. ``None`` selects
            ``observed`` in every input frame.
        fit_options: Shared native fitting options; ``None`` uses kernel defaults.

    Returns:
        List in input order containing a FitResult or the caught exception for
        each frame. ValueError (including NativeError), TypeError and PolarsError
        are retained as values, so callers must inspect every element.

    Raises:
        Exception: Errors outside the retained categories, including failures
            while advancing the input iterable, propagate to the caller.
    """
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
    """Predict each paired frame sequentially, preserving failed fits as values.

    Args:
        datasets: Iterable of pairs ``(fitted, frame)``. The first element is a
            FitResult or an existing Exception; the second is a Polars frame
            satisfying that fit's prediction contract.
        quantity: Shared ``"level"`` or ``"rate"`` output choice; defaults to level.
        covariate_rate_columns: Shared optional Logistic control-to-derivative-column
            mapping for rate predictions.

    Returns:
        Ordered list of prediction tables or exceptions. Existing fit exceptions
        are passed through unchanged. ValueError, TypeError and PolarsError from
        prediction are retained at the corresponding position.

    Raises:
        Exception: Errors outside the retained categories, malformed pair
            unpacking, and errors while advancing the iterable propagate.
    """
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
    """Classify observation times using sample-weighted nonnegative adoption rates.

    This historical categorization weights sampled times by their rate values;
    it does not integrate unequal time intervals or classify individual people.

    Args:
        data: Polars table with nonempty finite numeric ``time`` and rate columns,
            no nulls, strictly increasing times, nonnegative rates and positive
            total rate mass. Lazy input is collected and IPC copies buffers.
        rate_column: Name of the rate column; defaults to ``"adoption_rate"``.

    Returns:
        Input columns in the original order with an added string ``category``
        column containing the native historical adopter labels.

    Raises:
        ValueError: If time/rates violate the native contract, required columns
            are absent, or a ``category`` column already exists.
        TypeError: If data is not a supported Polars frame.
        polars.exceptions.PolarsError: If lazy collection or IPC processing fails.
    """
    output = _native.categorize_adopters_ipc(_ipc(data), rate_column)
    return pl.read_ipc(io.BytesIO(output))


def cumulative_sum(data: pl.Series) -> pl.Series:
    """Compute cumulative totals in the supplied sample order using Polars.

    Args:
        data: Polars Series with a dtype supported by ``Series.cum_sum``. The
            caller supplies observation order; no sorting or time inference occurs.

    Returns:
        Series of cumulative sums with Polars' dtype-promotion, null and
        non-finite-value semantics. This helper does not validate finite data.

    Raises:
        polars.exceptions.PolarsError: If the series dtype does not support summation.
        AttributeError: If the supplied object does not expose ``cum_sum``.
    """
    return data.cum_sum()


def rolling_average(data: pl.Series, window: int) -> pl.Series:
    """Compute trailing means, requiring a complete observation window.

    Polars computes the initial result. If a complete window of finite inputs
    produces a non-finite mean, a native numerically guarded calculation repairs
    that window. Null and non-finite input windows retain Polars' behaviour.

    Args:
        data: Numeric Polars Series in the desired sample order; no sorting occurs.
        window: Positive number of consecutive samples in each trailing window.

    Returns:
        Series of trailing means, with leading nulls until a full window exists.
        Windows lacking enough non-null observations also follow Polars semantics.

    Raises:
        ValueError: If ``window`` is less than one.
        NativeError: If a required native repair cannot be computed.
        TypeError: If the window or data cannot be processed with the supplied types.
        polars.exceptions.PolarsError: If the dtype does not support the operations.
    """
    if window < 1:
        raise ValueError("window must be positive")
    result = data.rolling_mean(window_size=window, min_samples=window)
    if not result.is_finite().all():
        # Only repair complete windows of finite samples. Leave Polars'
        # null and non-finite input behavior intact for other windows.
        finite = data.is_finite().fill_null(False)
        complete = finite.cast(pl.UInt32).rolling_sum(window, min_samples=window) == window
        if (complete & ~result.is_finite()).any():
            safe = data.fill_null(0).set(~finite, 0)
            values = execute("preprocess_series", method="rolling_average", window=window, values=safe.to_list())
            repaired = pl.Series(data.name, values["result"]["values"], dtype=result.dtype)
            return repaired.zip_with(complete.fill_null(False), result)
    return result


def aggregate_weekly(
    data: pl.DataFrame | pl.LazyFrame, value_columns: list[str], *, time_column: str = "time"
) -> pl.DataFrame:
    """Sum observations into 168-hour windows anchored at the earliest timestamp.

    Args:
        data: Polars frame with non-null Date or Datetime timestamps; lazy input
            is collected and observations are sorted before grouping.
        value_columns: Nonempty list of distinct numeric columns to sum.
        time_column: Timestamp column name; defaults to ``"time"``.

    Returns:
        Table of left-labelled weekly windows and value sums. Empty weekly windows
        are not inserted; the anchor is a data timestamp, not a calendar weekday.

    Raises:
        ValueError: If timestamps or value-column selections violate the
            ``aggregate_time_series`` validation rules.
        polars.exceptions.PolarsError: If columns are missing or grouping fails.
    """
    return aggregate_time_series(data, "W", value_columns, time_column=time_column)


def aggregate_time_series(
    data: pl.DataFrame | pl.LazyFrame, frequency: str, value_columns: list[str], *, time_column: str = "time"
) -> pl.DataFrame:
    """Aggregate numeric values by daily, weekly or calendar-month windows.

    The input is sorted by time and windows are left-closed. Daily and monthly
    results are upsampled between observed bounds and missing aggregate values
    are filled with zero. Weekly results keep only windows produced by grouping.

    Args:
        data: Polars DataFrame or LazyFrame; lazy input is collected before grouping.
        frequency: ``"D"`` for calendar days, ``"W"`` for fixed 168-hour windows
            anchored at the earliest observation, or ``"M"``/``"ME"`` for
            calendar-month windows relabelled to month end.
        value_columns: Nonempty list of distinct existing numeric columns to sum.
            Summation retains Polars' handling of null and non-finite values.
        time_column: Existing Date or Datetime column without nulls; defaults to
            ``"time"``. Numeric time axes are not accepted by this helper.

    Returns:
        Sorted DataFrame with the timestamp label and requested value sums.
        Monthly labels are shifted to the final date of their calendar month.

    Raises:
        ValueError: If frequency is unsupported, timestamps are not Date/Datetime
            or contain nulls, or value columns are empty, duplicated or nonnumeric.
        polars.exceptions.PolarsError: If named columns are missing or grouping,
            sorting or upsampling fails.
    """
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
    """Separate an evenly spaced numeric series into trend, season and residual.

    Samples are converted to JSON and processed by the native seasonal smoother.
    The caller must establish even spacing: a Series provides no timestamps to
    check. The result is a decomposition, not a forecast or uncertainty estimate.

    Args:
        data: Finite numeric Polars Series without nulls, ordered in time.
        period: Samples per seasonal cycle, at least two; the series must contain
            at least three complete cycles.
        trend_span: Trend smoothing window. Required unless ``historical=True``;
            historical mode derives an odd window from period and seasonal span
            when this is ``None``. Native window-length restrictions apply.
        robust: Whether to use robust reweighting; defaults to ``False``.
        historical: Whether to use historical STL local-fitting and reweighting
            rules; defaults to ``False``.
        inner_iterations: Inner smoothing passes, from 1 through 100; defaults to 5.
        outer_iterations: Robust reweighting passes, from 0 through 100. ``None``
            selects 5 when robust and 0 otherwise; positive values require robust mode.
        seasonal_span: Optional odd seasonal smoothing span of at least 3. Defaults
            to 7 in historical mode or at most 7 samples per phase otherwise.
        low_pass_span: Optional odd low-pass span of at least 3; the native default
            is the smallest odd number strictly greater than ``period``.
        seasonal_degree: Seasonal local polynomial degree, 0 or 1; defaults to 1.
        seasonal_jump: Positive spacing between fitted seasonal points; defaults to 1.
        trend_degree: Trend local polynomial degree, 0 or 1; defaults to 1.
        trend_jump: Positive spacing between fitted trend points; defaults to 1.
        low_pass_degree: Low-pass polynomial degree, 0 or 1; defaults to 1.
        low_pass_jump: Positive spacing between fitted low-pass points; defaults to 1.

    Returns:
        DataFrame with aligned ``trend``, ``seasonal`` and ``residual`` columns,
        one row per sample in the original order.

    Raises:
        NativeError: If sample counts, spans, iteration controls or numerical
            computations violate the native seasonal-decomposition contract.
        ValueError: If serialization encounters non-finite values.
        TypeError: If input values or options cannot be serialized as JSON.
        polars.exceptions.PolarsError: If native results cannot form a DataFrame.
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
    """Fit a no-intercept native SARIMA model and return in-sample predictions.

    The caller supplies evenly spaced, ordered samples; timestamps are not
    available to this wrapper. This function returns fitted values only, not
    coefficient estimates, future forecasts or prediction intervals.

    Args:
        data: Finite numeric Polars Series without nulls, in observation order.
        order: Nonnegative ordinary ``(p, d, q)`` autoregressive, differencing and
            moving-average orders, subject to native order and sample-size bounds.
        seasonal_order: Nonnegative ``(P, D, Q, period)`` seasonal orders; defaults
            to ``(0, 0, 0, 0)``. Seasonal differencing needs a period of at least two.
        historical: Include historical diffuse-initialization predictions when
            applicable. Defaults to ``False``, which leaves differencing warmup
            positions null in the returned series.

    Returns:
        Float64 Series with the input name and length, aligned to the original
        samples. Initial nulls reflect differencing warmup outside historical mode.

    Raises:
        NativeError: If orders, sample counts, stationarity constraints or native
            estimation cannot satisfy the supported SARIMA contract.
        ValueError: If serialization encounters non-finite values.
        TypeError: If data or order values are not JSON-compatible.
        polars.exceptions.PolarsError: If fitted output cannot form a numeric series.
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


def bootstrap_forecast_frame(result: dict[str, Any]) -> pl.DataFrame:
    """Flatten native bootstrap forecast arrays into time-by-series rows.

    Args:
        result: Native result payload, not its outer envelope, containing
            ``forecast.time``, ``columns``, ``baseline_rows`` and
            ``pointwise_intervals`` with matching dimensions. Missing interval
            estimates are represented by ``None``.

    Returns:
        DataFrame with ``time``, ``series``, ``baseline``, ``lower`` and ``upper``
        columns, ordered by input time then series. Numeric columns are Float64;
        missing interval bounds become null. No confidence level is inferred.

    Raises:
        KeyError: If the required forecast fields are absent.
        ValueError: If strict array lengths differ or interval pairs are malformed.
        TypeError: If the payload does not contain the expected iterable structures.
        polars.exceptions.PolarsError: If rows cannot be converted to the output schema.
    """
    forecast = result["forecast"]
    rows = []
    for time, baseline, intervals in zip(
        forecast["time"], forecast["baseline_rows"], forecast["pointwise_intervals"], strict=True
    ):
        for series, prediction, bounds in zip(forecast["columns"], baseline, intervals, strict=True):
            lower, upper = (None, None) if bounds is None else bounds
            rows.append((time, series, prediction, lower, upper))
    return pl.DataFrame(
        rows,
        schema={
            "time": pl.Float64,
            "series": pl.String,
            "baseline": pl.Float64,
            "lower": pl.Float64,
            "upper": pl.Float64,
        },
        orient="row",
    )


def treatment_contrast_frame(result: dict[str, Any]) -> pl.DataFrame:
    """Project a treatment-contrast result into the historical tabular schema.

    Args:
        result: Native result payload containing ``overall.estimate`` and a
            ``bootstrap`` mapping with interval bounds and draw count, or ``None``
            when bootstrap results are unavailable.

    Returns:
        One-row DataFrame with Float64 ``estimate``, ``ci_lower`` and ``ci_upper``
        and Int32 ``n_bootstrap``. An absent bootstrap produces null bounds and
        zero draws; this projection does not calculate new uncertainty estimates.

    Raises:
        KeyError: If required result or bootstrap fields are missing.
        TypeError: If the result does not have the expected mapping structure.
        polars.exceptions.PolarsError: If values cannot satisfy the output schema.
    """
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
    """Project native policy metadata into a one-row typed DataFrame.

    Args:
        result: Native result payload whose ``metadata`` contains intervention,
            model, treatment and outcome names and the confounder count.

    Returns:
        One-row table with string ``intervention_name``, ``model_name``,
        ``treatment_variable`` and ``outcome_variable``, and Int32
        ``n_confounders``. Other metadata fields are not part of this projection.

    Raises:
        KeyError: If the result lacks ``metadata``.
        TypeError: If the result has an incompatible structure.
        polars.exceptions.PolarsError: If metadata cannot satisfy the output schema.
    """
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


def staggered_att_polars(
    frame: pl.DataFrame | pl.LazyFrame,
    adoption_times: dict[str, float | None],
    **options: Any,
) -> dict[str, Any]:
    """Estimate staggered treatment effects from a native long-panel request.

    This wrapper selects the required columns and materializes them as Python
    lists before JSON serialization. A returned estimate does not establish the
    causal assumptions needed for interpretation, including valid comparison groups.

    Args:
        frame: Polars table with ``unit_ids``, numeric ``time`` and numeric
            ``outcomes`` columns. Lazy input is collected after column selection.
        adoption_times: Mapping from unit identifiers to treatment-adoption times
            in the panel's time units, or ``None`` for never-treated units.
        **options: Additional native ``staggered_long_panel_att`` options. They
            must not duplicate the selected columns or the wrapper's fixed fields.

    Returns:
        Native result payload for staggered long-panel average treatment effects,
        with the native diagnostics and aggregation fields retained.

    Raises:
        NativeError: If the panel, adoption map or options fail native validation.
        TypeError: If option names collide with fixed fields or values cannot be
            serialized as JSON.
        ValueError: If serialization encounters non-finite or circular values.
        polars.exceptions.PolarsError: If required columns are missing or collection fails.
    """
    panel = frame.select("unit_ids", "time", "outcomes")
    if isinstance(panel, pl.LazyFrame):
        panel = panel.collect()
    return execute(
        "summarize_causal",
        method="staggered_long_panel_att",
        **panel.to_dict(as_series=False),
        adoption_times=adoption_times,
        **options,
    )["result"]


def ingest_polars(
    frame: pl.DataFrame | pl.LazyFrame,
    kind: str,
    *,
    provenance: dict[str, Any],
    unit: str | None = None,
    directed: bool = False,
) -> dict[str, Any]:
    """Convert a Polars table into a native validated dataset artifact.

    Required columns depend on kind: adoption uses ``time`` and ``adoption`` with
    optional ``denominator``; substitution uses ``time`` and columns beginning
    with ``share``; competition uses ``time``, ``value``, ``unit_id`` and
    ``product_id``; policy timing uses ``event_times`` and ``event_effects`` with
    optional ``event_labels``; network edges use ``source`` and ``target`` with
    optional ``weight``. Other columns are not included in the artifact.

    Args:
        frame: Polars DataFrame or LazyFrame. Lazy input is collected; numeric
            columns are cast to Float64 and identifier columns to String, then
            copied into JSON-compatible arrays for native validation.
        kind: ``"adoption"``, ``"substitution"``, ``"competition"``,
            ``"policy_timing"`` or ``"network_edges"``.
        provenance: Native schema-1.0 provenance mapping, including explicit source,
            license, extraction time and transformation information. The kernel
            validates it and computes or verifies the content checksum.
        unit: Optional dataset unit label; omitted from the payload when ``None``.
        directed: Whether network edges are directed; defaults to ``False`` and
            has no effect on other dataset kinds.

    Returns:
        Native ingestion result with the validated dataset and checksum evidence.
        Substitution share columns retain input order, with labels derived from
        their suffixes. The wrapper does not establish the truth of supplied provenance.

    Raises:
        ValueError: If the kind is unsupported, required columns are missing,
            substitution has no share columns, or JSON values are non-finite.
        NativeError: If dataset content or provenance violates the native contract.
        TypeError: If provenance or data values are not JSON-compatible.
        polars.exceptions.PolarsError: If collection, selection or casting fails.
    """
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    dataset: dict[str, Any] = {"kind": kind, "schema_version": "1.0", "provenance": provenance}
    if unit is not None:
        dataset["unit"] = unit

    def column(name: str, dtype: type[pl.DataType]) -> list[Any]:
        """Require a named column, cast it, and materialize its values for JSON.

        Missing columns raise ``ValueError`` before native ingestion; invalid
        casts retain the original Polars exception and are not coerced to null.
        """
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
                name.removeprefix("share").lstrip("_") or f"product_{index}" for index, name in enumerate(shares)
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
        )
        if "event_labels" in frame.columns:
            dataset["event_labels"] = column("event_labels", pl.String)
    elif kind == "network_edges":
        dataset.update(source=column("source", pl.String), target=column("target", pl.String), directed=directed)
        if "weight" in frame.columns:
            dataset["weight"] = column("weight", pl.Float64)
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
    """Read a local table and pass it through native dataset ingestion.

    CSV identifier columns are read as strings for competition, policy-timing and
    network-edge datasets so numeric-looking identifiers are not intentionally
    converted to numbers. A ``load:<filename>`` transform and the path and format
    are added to provenance. The path is therefore retained in the output artifact;
    use suitable paths and provenance when the artifact will be shared.

    Args:
        path: Existing local file path; URLs and directories are not supported.
        kind: Native dataset kind accepted by ``ingest_polars``.
        provenance: Native provenance mapping. Existing transform steps and extra
            fields are preserved except for the added path/format fields.
        format: Explicit format or, by default, the path's suffix. Accepted values
            are ``csv``/``txt``, ``parquet``/``pq``, ``arrow``/``feather``/``ipc``,
            and ``stream``/``ipc_stream``; leading dots and case are normalized.
            Arrow/file aliases inspect the leading IPC file magic to select the reader.
        unit: Optional dataset unit passed unchanged to ``ingest_polars``.
        directed: Network-edge direction flag, defaulting to ``False``; ignored
            for other dataset kinds.

    Returns:
        Native validated, checksummed ingestion result with augmented provenance.
        The full table is read locally before native validation.

    Raises:
        FileNotFoundError: If the supplied path is not an existing regular file.
        ValueError: If the format or dataset kind is unsupported, columns are
            missing, or JSON serialization encounters non-finite values.
        NativeError: If dataset or provenance validation fails.
        OSError: If the file cannot be opened or read.
        TypeError: If provenance fields or serialized values have incompatible types.
        polars.exceptions.PolarsError: If parsing, schema conversion or ingestion casting fails.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    file_format = (format or path.suffix).lstrip(".").lower()
    if file_format in {"csv", "txt"}:
        identifier_columns = {
            "competition": ("unit_id", "product_id"),
            "policy_timing": ("event_labels",),
            "network_edges": ("source", "target"),
        }.get(kind, ())
        frame = pl.read_csv(path, schema_overrides={name: pl.String for name in identifier_columns})
    elif file_format in {"parquet", "pq"}:
        frame = pl.read_parquet(path)
    elif file_format in {"arrow", "feather", "ipc"}:
        with path.open("rb") as source:
            is_file = source.read(6) == b"ARROW1"
        frame = pl.read_ipc(path) if is_file else pl.read_ipc_stream(path)
    elif file_format in {"stream", "ipc_stream"}:
        frame = pl.read_ipc_stream(path)
    else:
        raise ValueError(f"unsupported local format: {file_format}")
    file_provenance = {
        **provenance,
        "transform_steps": [*provenance.get("transform_steps", []), f"load:{path.name}"],
        "extra": {**provenance.get("extra", {}), "path": str(path), "format": file_format},
    }
    return ingest_polars(frame, kind, provenance=file_provenance, unit=unit, directed=directed)
