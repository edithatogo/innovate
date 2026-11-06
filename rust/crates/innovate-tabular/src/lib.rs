//! Polars-owned tabular contracts and Arrow IPC exchange. IPC currently copies.
#![forbid(unsafe_code)]
use innovate_core::{KernelBinding, KernelRequest, json};
use polars::prelude::*;
use serde_json::Value;
use std::io::Cursor;

fn numbers(frame: &DataFrame, name: &str) -> PolarsResult<Vec<f64>> {
    let column = frame.column(name)?;
    if !column.dtype().is_numeric() {
        polars_bail!(SchemaMismatch: "{name} must be numeric");
    }
    let column = column.cast(&DataType::Float64)?;
    let array = column.f64()?;
    if array.is_empty() || array.null_count() != 0 {
        polars_bail!(ComputeError: "{name} must be nonempty without nulls");
    }
    let values: Vec<f64> = array.into_no_null_iter().collect();
    if values.iter().any(|v| !v.is_finite()) {
        polars_bail!(ComputeError: "{name} must be finite");
    }
    Ok(values)
}

fn inputs(frame: &DataFrame, observed: bool) -> PolarsResult<Value> {
    let time = numbers(frame, "time")?;
    if time.windows(2).any(|p| p[1] <= p[0]) {
        polars_bail!(ComputeError: "time must be strictly increasing");
    }
    let mut input = json!({"time":time});
    if observed {
        input["observed"] = json!(numbers(frame, "observed")?);
    }
    Ok(input)
}

pub fn predict(mut frame: DataFrame, model: &str, parameters: Value) -> PolarsResult<DataFrame> {
    if frame.column("prediction").is_ok() {
        polars_bail!(Duplicate: "prediction column already exists");
    }
    let state = json!({"model_key":model, "parameters":parameters, "constructor_kwargs":{}, "predict_kwargs":{}});
    let request = KernelRequest::predict_model(
        model,
        json!({"inputs":inputs(&frame,false)?, "state":state}),
    );
    let response = KernelBinding::new()
        .invoke(&request)
        .map_err(|e| polars_err!(ComputeError: "{e}"))?;
    let result = response
        .result
        .ok_or_else(|| polars_err!(ComputeError: "missing native result"))?;
    let values = if let Some(values) = result["values"].as_array() {
        values
            .iter()
            .map(|v| v.as_f64())
            .collect::<Option<Vec<_>>>()
    } else {
        result["rows"].as_array().and_then(|rows| {
            rows.iter()
                .map(|r| r[0].as_f64())
                .collect::<Option<Vec<_>>>()
        })
    }
    .ok_or_else(|| polars_err!(ComputeError: "native predictions are not finite scalar values"))?;
    if values.len() != frame.height() || values.iter().any(|v| !v.is_finite()) {
        polars_bail!(ComputeError: "native prediction shape or finite-value violation");
    }
    frame.with_column(Series::new("prediction".into(), values).into())?;
    Ok(frame)
}

pub fn predict_ipc(data: &[u8], model: &str, parameters: Value) -> PolarsResult<Vec<u8>> {
    let frame = IpcReader::new(Cursor::new(data)).finish()?;
    let mut output = predict(frame, model, parameters)?;
    let mut bytes = Vec::new();
    IpcWriter::new(&mut bytes).finish(&mut output)?;
    Ok(bytes)
}

pub fn fit_ipc(data: &[u8], model: &str) -> PolarsResult<String> {
    let frame = IpcReader::new(Cursor::new(data)).finish()?;
    let request = KernelRequest::fit_model(model, json!({"inputs": inputs(&frame,true)?}));
    Ok(innovate_core::execute_json(
        &serde_json::to_string(&request).expect("serializable request"),
    ))
}
