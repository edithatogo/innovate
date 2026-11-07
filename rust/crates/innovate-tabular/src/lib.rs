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

fn model_inputs(frame: &DataFrame, observed: &[String], kwargs: &Value) -> PolarsResult<Value> {
    let mut input = inputs(frame, observed)?;
    if let Some(names) = kwargs.get("covariates") {
        let names: Vec<String> = serde_json::from_value(names.clone())
            .map_err(|_| polars_err!(SchemaMismatch:"covariates must be column names"))?;
        if !names.is_empty() {
            let mut values = serde_json::Map::new();
            for name in names {
                values.insert(name.clone(), json!(numbers(frame, &name)?));
            }
            input["covariates"] = Value::Object(values);
        }
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
        .any(|k| !["model_options", "quantity", "covariate_rate_columns"].contains(&k.as_str()))
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
    let state = json!({"model_key":model, "parameters":parameters, "constructor_kwargs":kwargs.clone(), "predict_kwargs":{}});
    let mut input = model_inputs(&frame, &[], &kwargs)?;
    if let Some(columns) = options.get("covariate_rate_columns") {
        if model != "logistic" || quantity != "rate" || !input["covariates"].is_object() {
            polars_bail!(SchemaMismatch:"covariate_rate_columns requires Logistic rate predictions with declared controls");
        }
        let columns = columns.as_object()
            .ok_or_else(|| polars_err!(SchemaMismatch:"covariate_rate_columns must map controls to column names"))?;
        let controls = input["covariates"]
            .as_object()
            .expect("declared controls checked above");
        if columns.len() != controls.len()
            || columns.keys().any(|name| !controls.contains_key(name))
        {
            polars_bail!(SchemaMismatch:"covariate_rate_columns must match declared controls");
        }
        let mut rates = serde_json::Map::new();
        for (name, column) in columns {
            let column = column.as_str().ok_or_else(
                || polars_err!(SchemaMismatch:"covariate rate column names must be strings"),
            )?;
            rates.insert(name.clone(), json!(numbers(&frame, column)?));
        }
        input["covariate_rates"] = Value::Object(rates);
    }
    let request = KernelRequest::predict_model(model, json!({"inputs":input, "state":state}));
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
        json!({"inputs": model_inputs(&frame,&columns,options.get("model_options").unwrap_or(&json!({})))?,
        "constructor_kwargs":options.get("model_options").cloned().unwrap_or_else(||json!({})),
        "fit_options":options.get("fit_options").cloned().unwrap_or_else(||json!({}))}),
    );
    Ok(innovate_core::execute_json(
        &serde_json::to_string(&request).expect("serializable request"),
    ))
}

/// Assigns historical adopter categories using sample-weighted adoption times.
pub fn categorize_adopters(mut frame: DataFrame, rate_column: &str) -> PolarsResult<DataFrame> {
    let time = numbers(&frame, "time")?;
    let rates = numbers(&frame, rate_column)?;
    let categories = innovate_core::classify_adopters(&time, &rates)
        .map_err(|error| polars_err!(ComputeError: "{error}"))?;
    if frame.column("category").is_ok() {
        polars_bail!(Duplicate: "category column already exists");
    }
    frame.with_column(Series::new("category".into(), categories).into())?;
    Ok(frame)
}

pub fn categorize_adopters_ipc(data: &[u8], rate_column: &str) -> PolarsResult<Vec<u8>> {
    let frame = IpcReader::new(Cursor::new(data)).finish()?;
    let mut output = categorize_adopters(frame, rate_column)?;
    let mut bytes = Vec::new();
    IpcWriter::new(&mut bytes).finish(&mut output)?;
    Ok(bytes)
}
