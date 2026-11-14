//! CPython transport for the Rust-owned kernel and Polars/Arrow operations.
//!
//! Python arguments are copied or decoded into native inputs. Computational work
//! runs while detached from the Python interpreter; there is no Python numerical
//! fallback. JSON-envelope errors and raised transport errors are distinct.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Rewrite Arrow IPC file or stream bytes as an interoperable IPC file.
///
/// Args:
///     data: Complete Arrow IPC file or stream bytes to decode through Polars.
/// Returns:
///     Newly allocated IPC file bytes, retaining schema metadata and compatible
///     field annotations. Conversion copies data and may change the byte encoding.
/// Raises:
///     ValueError: If Arrow decoding, schema conversion or encoding fails.
///     TypeError: If Python arguments cannot be converted to the declared types.
#[pyfunction]
fn export_ipc_file(py: Python<'_>, data: Vec<u8>) -> PyResult<Py<PyBytes>> {
    let bytes = py
        .detach(move || innovate_tabular::export_ipc_file(&data).map_err(|error| error.to_string()))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Rewrite Arrow IPC file or stream bytes as a stream with leading schema metadata.
///
/// Args:
///     data: Complete Arrow IPC file or stream bytes to decode through Polars.
/// Returns:
///     Newly allocated stream bytes retaining custom schema metadata and compatible
///     field annotations. This is a copying conversion, not a zero-copy view.
/// Raises:
///     ValueError: If Arrow decoding, schema conversion or encoding fails.
///     TypeError: If Python arguments cannot be converted to the declared types.
#[pyfunction]
fn export_ipc_stream(py: Python<'_>, data: Vec<u8>) -> PyResult<Py<PyBytes>> {
    let bytes = py
        .detach(move || {
            innovate_tabular::export_ipc_stream(&data).map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Execute one serialized native request and return its serialized response envelope.
///
/// Args:
///     request: JSON text containing schema_version, operation and its payload,
///         with model_key where the selected operation requires a model.
/// Returns:
///     JSON response text. Inspect its error member before using result;
///     malformed JSON and unsupported native requests produce structured errors.
/// Raises:
///     TypeError: If request cannot be converted to a Python string argument.
///
/// No scientific fallback is attempted. A returned string alone does not establish
/// that a fit, prediction or simulation completed successfully.
#[pyfunction]
fn execute(py: Python<'_>, request: String) -> String {
    py.detach(move || innovate_core::execute_json(&request))
}

/// Append native predictions or signed rates to an Arrow IPC input table.
///
/// Args:
///     data: IPC file or stream bytes with finite numeric time and required controls.
///     model: Native model identity, such as bass or logistic.
///     parameters: JSON text encoding the model's parameter object.
///     options: Optional JSON object with model_options, quantity and
///         covariate_rate_columns; omitted options use level predictions.
/// Returns:
///     New IPC file bytes with the original columns plus prediction or rate columns.
///     Multivariate outputs receive numbered columns; rates use the input time unit.
/// Raises:
///     ValueError: If JSON, IPC, column contracts, native execution or output shape
///         is invalid, including a collision with a generated column name.
///     TypeError: If Python arguments cannot be converted to the declared types.
#[pyfunction]
#[pyo3(signature=(data, model, parameters, options=None))]
fn predict_ipc(
    py: Python<'_>,
    data: Vec<u8>,
    model: String,
    parameters: String,
    options: Option<String>,
) -> PyResult<Py<PyBytes>> {
    let output = py
        .detach(move || {
            let parameters = serde_json::from_str(&parameters).map_err(|e| e.to_string())?;
            let options = serde_json::from_str(options.as_deref().unwrap_or("{}"))
                .map_err(|e| e.to_string())?;
            innovate_tabular::predict_ipc_with_options(&data, &model, parameters, options)
                .map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &output).unbind())
}

/// Fit a Rust-native model from an Arrow IPC table and return a JSON envelope.
///
/// Args:
///     data: IPC file or stream bytes with time and selected observation columns.
///     model: Native model identity for the fitting operation.
///     options: Optional JSON object with model_options, observed_columns and
///         fit_options; observations default to the observed column.
/// Returns:
///     JSON response text carrying fitted state or a structured native fit error.
///     Inspect error before consuming result; transport success is not fit success.
/// Raises:
///     ValueError: If option JSON, IPC data or the tabular input contract is invalid.
///     TypeError: If Python arguments cannot be converted to the declared types.
#[pyfunction]
#[pyo3(signature=(data, model, options=None))]
fn fit_ipc(
    py: Python<'_>,
    data: Vec<u8>,
    model: String,
    options: Option<String>,
) -> PyResult<String> {
    py.detach(move || {
        let options =
            serde_json::from_str(options.as_deref().unwrap_or("{}")).map_err(|e| e.to_string())?;
        innovate_tabular::fit_ipc_with_options(&data, &model, options).map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

/// Append historical adopter categories using aligned nonnegative sample weights.
///
/// Args:
///     data: IPC file or stream bytes with numeric time and the selected rate column.
///     rate_column: Column containing nonnegative weights with positive total.
/// Returns:
///     New IPC file bytes containing an additional category string column.
///     Weights apply per sample without time-step-width multiplication.
/// Raises:
///     ValueError: If data are invalid, times are not strictly increasing, weights
///         are negative/all zero, or a category column already exists.
///     TypeError: If Python arguments cannot be converted to the declared types.
#[pyfunction]
fn categorize_adopters_ipc(
    py: Python<'_>,
    data: Vec<u8>,
    rate_column: String,
) -> PyResult<Py<PyBytes>> {
    let output = py
        .detach(move || {
            innovate_tabular::categorize_adopters_ipc(&data, &rate_column)
                .map_err(|e| e.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &output).unbind())
}

/// Execute a scenario grid using one shared time column from an IPC table.
///
/// Args:
///     data: IPC file or stream bytes with a finite, strictly increasing time column.
///     payload: JSON text whose scenarios contain simulation objects. Individual
///         simulation inputs must not already provide time or t.
/// Returns:
///     JSON grid response text. Native failures remain in its error fields and
///     must be inspected before treating any execution as successful.
/// Raises:
///     ValueError: If JSON/IPC decoding fails or the shared-time injection contract
///         is violated by malformed scenarios or conflicting time inputs.
///     TypeError: If Python arguments cannot be converted to the declared types.
#[pyfunction]
fn scenario_grid_ipc(py: Python<'_>, data: Vec<u8>, payload: String) -> PyResult<String> {
    py.detach(move || {
        let payload = serde_json::from_str(&payload).map_err(|e| e.to_string())?;
        innovate_tabular::scenario_grid_ipc(&data, payload).map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

/// Project ordered scenario execution rows into an Arrow IPC file.
///
/// Args:
///     response: JSON text of a successful execute_scenario_grid envelope.
/// Returns:
///     IPC file bytes with scenario_id, model_key, successful and execution_json.
///     Nested results/errors remain JSON; the success flag is not scientific validation.
/// Raises:
///     ValueError: If JSON decoding, envelope/row structure or Arrow encoding fails.
///     TypeError: If response cannot be converted to a Python string argument.
#[pyfunction]
fn scenario_executions_ipc(py: Python<'_>, response: String) -> PyResult<Py<PyBytes>> {
    let bytes = py
        .detach(move || {
            let response = serde_json::from_str(&response).map_err(|error| error.to_string())?;
            innovate_tabular::scenario_executions_ipc(&response).map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Project ordered pair and ensemble comparison rows into an Arrow IPC file.
///
/// Args:
///     response: JSON text of a successful execute_scenario_grid envelope.
/// Returns:
///     IPC file bytes with comparison_index, kind, successful and comparison_json.
///     Nested outcomes remain JSON; an absent row error is not scientific validation.
/// Raises:
///     ValueError: If JSON decoding, comparison structure or Arrow encoding fails.
///     TypeError: If response cannot be converted to a Python string argument.
#[pyfunction]
fn scenario_comparisons_ipc(py: Python<'_>, response: String) -> PyResult<Py<PyBytes>> {
    let bytes = py
        .detach(move || {
            let response = serde_json::from_str(&response).map_err(|error| error.to_string())?;
            innovate_tabular::scenario_comparisons_ipc(&response).map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Validate and flatten paired Bass strategy/draw trajectories into Arrow IPC.
///
/// Args:
///     payload: JSON text with at least two paired strategies, matching ordered
///         draw IDs, one shared time grid and parameter names/units for p, q and m.
/// Returns:
///     IPC file bytes ordered by strategy, draw and time. Provenance, time units,
///     optional perspective and original parameter arrays are retained in columns.
/// Raises:
///     ValueError: If JSON decoding, pairing/value contracts or Arrow encoding fails.
///     TypeError: If payload cannot be converted to a Python string argument.
///
/// Structural validation does not establish causal comparability or economic validity.
#[pyfunction]
fn paired_outcomes_ipc(py: Python<'_>, payload: String) -> PyResult<Py<PyBytes>> {
    let bytes = py
        .detach(move || {
            let payload = serde_json::from_str(&payload).map_err(|error| error.to_string())?;
            innovate_tabular::paired_outcomes_ipc(&payload).map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

/// Native Rust transport used by the supported Python binding.
///
/// Functions exchange JSON envelopes and Arrow IPC bytes. Returned JSON must be
/// checked for structured errors; no Python numerical fallback is available.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(execute, module)?)?;
    module.add_function(wrap_pyfunction!(export_ipc_file, module)?)?;
    module.add_function(wrap_pyfunction!(export_ipc_stream, module)?)?;
    module.add_function(wrap_pyfunction!(predict_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(fit_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(categorize_adopters_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(scenario_grid_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(scenario_executions_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(scenario_comparisons_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(paired_outcomes_ipc, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
