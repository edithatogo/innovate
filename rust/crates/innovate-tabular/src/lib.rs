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

fn inputs(frame: &DataFrame, observed: &[String]) -> PolarsResult<Value> {
    let time = numbers(frame, "time")?;
    if time.windows(2).any(|p| p[1] <= p[0]) {
        polars_bail!(ComputeError: "time must be strictly increasing");
    }
    let mut input = json!({"time":time});
    if !observed.is_empty() {
        if observed
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != observed.len()
        {
            polars_bail!(Duplicate: "observation columns must be unique");
        }
        let columns = observed
            .iter()
            .map(|name| numbers(frame, name))
            .collect::<PolarsResult<Vec<_>>>()?;
        input["observed"] = if columns.len() == 1 {
            json!(columns[0])
        } else {
            json!(
                (0..frame.height())
                    .map(|i| columns.iter().map(|c| c[i]).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            )
        };
    }
    Ok(input)
}

pub fn predict(frame: DataFrame, model: &str, parameters: Value) -> PolarsResult<DataFrame> {
    predict_with_options(frame, model, parameters, json!({}))
}

pub fn predict_with_options(
    mut frame: DataFrame,
    model: &str,
    parameters: Value,
    options: Value,
) -> PolarsResult<DataFrame> {
    let options = options
        .as_object()
        .ok_or_else(|| polars_err!(SchemaMismatch:"options must be an object"))?;
    if options
        .keys()
        .any(|k| !["model_options", "quantity"].contains(&k.as_str()))
    {
        polars_bail!(SchemaMismatch:"unknown prediction option");
    }
    let quantity = options
        .get("quantity")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| polars_err!(SchemaMismatch:"quantity must be a string"))
        })
        .transpose()?
        .unwrap_or("level");
    if !["level", "rate"].contains(&quantity) {
        polars_bail!(SchemaMismatch:"quantity must be level or rate; signed rates are not gross incidence");
    }
    let kwargs = options
        .get("model_options")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let state = json!({"model_key":model, "parameters":parameters, "constructor_kwargs":kwargs, "predict_kwargs":{}});
    let request =
        KernelRequest::predict_model(model, json!({"inputs":inputs(&frame,&[])?, "state":state}));
    let response = KernelBinding::new()
        .invoke(&request)
        .map_err(|e| polars_err!(ComputeError: "{e}"))?;
    let result = response
        .result
        .ok_or_else(|| polars_err!(ComputeError: "missing native result"))?;
    let rows: Vec<Vec<f64>> = if quantity == "level" && result["values"].is_array() {
        result["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().map(|x| vec![x]))
            .collect::<Option<_>>()
            .ok_or_else(|| polars_err!(ComputeError:"native predictions must be numeric"))?
    } else {
        let key = if quantity == "level" { "rows" } else { "rates" };
        result[key].as_array().and_then(|rows|rows.iter().map(|r|r.as_array().and_then(|r|r.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())).collect::<Option<Vec<_>>>())
            .ok_or_else(||polars_err!(ComputeError:"requested native output is unavailable or not numeric"))?
    };
    let width = rows.first().map_or(0, Vec::len);
    if rows.len() != frame.height()
        || width == 0
        || rows
            .iter()
            .any(|r| r.len() != width || r.iter().any(|v| !v.is_finite()))
    {
        polars_bail!(ComputeError: "native prediction shape or finite-value violation");
    }
    let prefix = if quantity == "level" {
        "prediction"
    } else {
        "rate"
    };
    let names: Vec<String> = (0..width)
        .map(|i| {
            if width == 1 {
                prefix.to_string()
            } else {
                format!("{prefix}_{}", i + 1)
            }
        })
        .collect();
    for name in &names {
        if frame.column(name).is_ok() {
            polars_bail!(Duplicate:"{name} column already exists");
        }
    }
    for (j, name) in names.into_iter().enumerate() {
        let values: Vec<_> = rows.iter().map(|r| r[j]).collect();
        frame.with_column(Series::new(name.into(), values).into())?;
    }
    Ok(frame)
}

pub fn predict_ipc(data: &[u8], model: &str, parameters: Value) -> PolarsResult<Vec<u8>> {
    predict_ipc_with_options(data, model, parameters, json!({}))
}

pub fn predict_ipc_with_options(
    data: &[u8],
    model: &str,
    parameters: Value,
    options: Value,
) -> PolarsResult<Vec<u8>> {
    let frame = IpcReader::new(Cursor::new(data)).finish()?;
    let mut output = predict_with_options(frame, model, parameters, options)?;
    let mut bytes = Vec::new();
    IpcWriter::new(&mut bytes).finish(&mut output)?;
    Ok(bytes)
}

pub fn fit_ipc(data: &[u8], model: &str) -> PolarsResult<String> {
    fit_ipc_with_options(data, model, json!({}))
}

pub fn fit_ipc_with_options(data: &[u8], model: &str, options: Value) -> PolarsResult<String> {
    let frame = IpcReader::new(Cursor::new(data)).finish()?;
    let options = options
        .as_object()
        .ok_or_else(|| polars_err!(SchemaMismatch:"options must be an object"))?;
    if options
        .keys()
        .any(|k| !["model_options", "observed_columns", "fit_options"].contains(&k.as_str()))
    {
        polars_bail!(SchemaMismatch:"unknown fitting option");
    }
    let columns: Vec<String> = if let Some(columns) = options.get("observed_columns") {
        serde_json::from_value(columns.clone())
            .map_err(|_| polars_err!(SchemaMismatch:"observed_columns must be strings"))?
    } else {
        vec!["observed".into()]
    };
    if columns.is_empty() {
        polars_bail!(SchemaMismatch:"at least one observed column is required");
    }
    let request = KernelRequest::fit_model(
        model,
        json!({"inputs": inputs(&frame,&columns)?,
        "constructor_kwargs":options.get("model_options").cloned().unwrap_or_else(||json!({})),
        "fit_options":options.get("fit_options").cloned().unwrap_or_else(||json!({}))}),
    );
    Ok(innovate_core::execute_json(
        &serde_json::to_string(&request).expect("serializable request"),
    ))
}
