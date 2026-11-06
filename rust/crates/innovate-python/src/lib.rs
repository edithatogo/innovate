use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyfunction]
fn execute(py: Python<'_>, request: String) -> String {
    py.detach(move || innovate_core::execute_json(&request))
}

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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(execute, module)?)?;
    module.add_function(wrap_pyfunction!(predict_ipc, module)?)?;
    module.add_function(wrap_pyfunction!(fit_ipc, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
