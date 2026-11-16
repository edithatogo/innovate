//! Polars-owned tabular contracts and Arrow IPC exchange. IPC currently copies.
#![forbid(unsafe_code)]
use innovate_core::{KernelBinding, KernelRequest, json};
use polars_arrow::datatypes::{ArrowDataType, Metadata};
use polars_arrow::io::ipc::read::{
    FileReader, StreamReader, StreamState, read_file_metadata, read_stream_metadata,
};
use polars_arrow::io::ipc::write::{FileWriter, StreamWriter, WriteOptions};
use polars_core::prelude::*;
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

/// Append native level predictions to a frame using default prediction options.
///
/// `frame` supplies a nonempty finite, strictly increasing numeric `time` column;
/// `model` and `parameters` identify the native model and its parameter object.
/// Existing columns remain, and output columns are named `prediction` or
/// `prediction_1`, `prediction_2`, etc. for multivariate results.
///
/// # Errors
/// Returns schema/compute errors for invalid input, unsupported model parameters,
/// unavailable native output or collisions with generated output column names.
pub fn predict(frame: DataFrame, model: &str, parameters: Value) -> PolarsResult<DataFrame> {
    predict_with_options(frame, model, parameters, json!({}))
}

/// Append native level or signed-rate predictions using explicit tabular options.
///
/// `options` accepts only `model_options`, `quantity` (`level` or `rate`) and
/// `covariate_rate_columns`. The latter maps every declared Logistic covariate to
/// its derivative column when requesting rates. Input time/covariate columns
/// must be nonempty, finite and numeric; time must be strictly increasing.
///
/// Level output uses `prediction` columns; rate output uses `rate` columns,
/// numbered from one for multivariate results. Rates are derivatives in the
/// supplied time unit and may be signed; they are not gross incidence.
///
/// # Errors
/// Rejects unknown/malformed options, invalid columns, native execution failures,
/// nonfinite/misaligned output and collisions with generated column names.
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

fn read_ipc_with_metadata(
    data: &[u8],
) -> PolarsResult<(DataFrame, ArrowSchema, Option<std::sync::Arc<Metadata>>)> {
    let mut cursor = Cursor::new(data);
    if data.starts_with(b"ARROW1") {
        let metadata = read_file_metadata(&mut cursor)?;
        let schema = metadata.schema.as_ref().clone();
        let custom = metadata.custom_schema_metadata.clone();
        let polars_schema = Schema::from_arrow_schema(&schema);
        let reader = FileReader::new(cursor, metadata, None, None);
        let mut frame = DataFrame::empty_with_schema(&polars_schema);
        for batch in reader {
            frame.append_record_batch(batch?)?;
        }
        Ok((frame, schema, custom))
    } else {
        let metadata = read_stream_metadata(&mut cursor)?;
        let schema = metadata.schema.clone();
        let custom = metadata
            .custom_schema_metadata
            .clone()
            .map(std::sync::Arc::new);
        let polars_schema = Schema::from_arrow_schema(&schema);
        let mut reader = StreamReader::new(cursor, metadata, None);
        let mut frame = DataFrame::empty_with_schema(&polars_schema);
        for state in &mut reader {
            if let StreamState::Some(batch) = state? {
                frame.append_record_batch(batch)?;
            }
        }
        Ok((frame, schema, custom))
    }
}

fn read_ipc(data: &[u8]) -> PolarsResult<DataFrame> {
    Ok(read_ipc_with_metadata(data)?.0)
}

/// Rewrite Arrow IPC file or stream bytes as interoperable IPC file bytes.
///
/// Preserves custom schema metadata and compatible field annotations while
/// materializing the data through Polars. This conversion copies data and does
/// not promise byte-for-byte identity with the input.
///
/// # Errors
/// Returns the underlying Arrow/Polars read, schema or write failure.
pub fn export_ipc_file(data: &[u8]) -> PolarsResult<Vec<u8>> {
    rewrite_ipc(data, false)
}

/// Rewrite Arrow IPC file or stream bytes as a stream with leading schema metadata.
///
/// Preserves custom schema metadata and compatible field annotations using the
/// same copying conversion as [`export_ipc_file`].
///
/// # Errors
/// Returns the underlying Arrow/Polars read, schema or write failure.
pub fn export_ipc_stream(data: &[u8]) -> PolarsResult<Vec<u8>> {
    rewrite_ipc(data, true)
}

fn rewrite_ipc(data: &[u8], stream: bool) -> PolarsResult<Vec<u8>> {
    let (mut frame, input_schema, metadata) = read_ipc_with_metadata(data)?;
    let compat = CompatLevel::oldest();
    let mut schema = frame.schema().to_arrow(compat);
    for (name, field) in schema.iter_mut() {
        if let Some(input) = input_schema.get(name.as_str()) {
            preserve_ipc_annotations(input, field);
        }
    }
    frame.align_chunks_par();
    let mut bytes = Vec::new();
    let options = WriteOptions { compression: None };
    if stream {
        let mut writer = StreamWriter::new(&mut bytes, options);
        if let Some(metadata) = metadata {
            writer.set_custom_schema_metadata(metadata);
        }
        writer.start(&schema, None)?;
        for batch in frame.iter_chunks(compat, true) {
            writer.write(&batch, None)?;
        }
        writer.finish()?;
    } else {
        let mut writer = FileWriter::new(&mut bytes, schema.into(), None, options);
        if let Some(metadata) = metadata {
            writer.set_custom_schema_metadata(metadata);
        }
        writer.start()?;
        for batch in frame.iter_chunks(compat, true) {
            writer.write(&batch, None)?;
        }
        writer.finish()?;
    }
    Ok(bytes)
}

fn preserve_ipc_annotations(input: &ArrowField, output: &mut ArrowField) -> bool {
    use ArrowDataType::{
        Binary, BinaryView, FixedSizeList, LargeBinary, LargeList, LargeUtf8, List, Struct, Utf8,
        Utf8View,
    };

    let same_logical_type = input.dtype == output.dtype
        || matches!(
            (&input.dtype, &output.dtype),
            (Utf8 | LargeUtf8 | Utf8View, Utf8 | LargeUtf8 | Utf8View)
                | (
                    Binary | LargeBinary | BinaryView,
                    Binary | LargeBinary | BinaryView
                )
        );
    let compatible_children = match (&input.dtype, &mut output.dtype) {
        (List(input) | LargeList(input), List(output) | LargeList(output)) => {
            preserve_ipc_annotations(input, output)
        }
        (FixedSizeList(input, a), FixedSizeList(output, b)) if a == b => {
            preserve_ipc_annotations(input, output)
        }
        (Struct(input), Struct(output)) => {
            let mut compatible = input.len() == output.len();
            for field in output {
                compatible &= input
                    .iter()
                    .find(|source| source.name == field.name)
                    .is_some_and(|source| preserve_ipc_annotations(source, field));
            }
            compatible
        }
        _ => false,
    };
    let compatible = same_logical_type || compatible_children;
    if compatible && input.metadata.is_some() {
        output.metadata = input.metadata.clone();
    }
    compatible
}

/// Read IPC file/stream bytes, append default level predictions, and return an IPC file.
///
/// Input and output are copied. See [`predict`] for column names and model units.
///
/// # Errors
/// Propagates IPC decoding/encoding errors and all validation errors from [`predict`].
pub fn predict_ipc(data: &[u8], model: &str, parameters: Value) -> PolarsResult<Vec<u8>> {
    predict_ipc_with_options(data, model, parameters, json!({}))
}

/// Predict from IPC file/stream bytes with the options accepted by [`predict_with_options`].
///
/// Returns a newly encoded IPC file containing original and prediction/rate columns.
/// This operation does not promise to retain arbitrary input schema metadata.
///
/// # Errors
/// Propagates IPC decoding/encoding failures and tabular/native prediction errors.
pub fn predict_ipc_with_options(
    data: &[u8],
    model: &str,
    parameters: Value,
    options: Value,
) -> PolarsResult<Vec<u8>> {
    let frame = read_ipc(data)?;
    let mut output = predict_with_options(frame, model, parameters, options)?;
    write_ipc_file(&mut output)
}

/// Fit a native model from IPC bytes using `time` and `observed` columns.
///
/// Returns a serialized native JSON envelope. The caller must check its `error`
/// member; an `Ok` string is not by itself proof of a successful scientific fit.
///
/// # Errors
/// Returns a Polars error for invalid IPC or tabular inputs. Native fit failures
/// are represented in the returned JSON envelope.
pub fn fit_ipc(data: &[u8], model: &str) -> PolarsResult<String> {
    fit_ipc_with_options(data, model, json!({}))
}

/// Fit a native model from IPC bytes with explicit column/model/fitter options.
///
/// `options` accepts only `model_options`, `observed_columns` and `fit_options`.
/// Observations default to the single `observed` column; multiple columns become
/// one row per time point in a multivariate payload. Time must be finite and
/// strictly increasing, and observation/covariate columns must be finite numeric data.
///
/// # Errors
/// Malformed options or input tables return a Polars error. Native validation and
/// fit failures are encoded in the returned JSON string's `error` member.
pub fn fit_ipc_with_options(data: &[u8], model: &str, options: Value) -> PolarsResult<String> {
    let frame = read_ipc(data)?;
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
///
/// Reads numeric `time` and `rate_column`, then appends a string `category` column.
/// Weights are used per sample, without time-step-width multiplication; see
/// [`innovate_core::classify_adopters`] for category boundaries and interpretation.
///
/// # Errors
/// Rejects invalid/negative/nonfinite weights, non-increasing time, zero total
/// weight and a pre-existing `category` column, alongside Polars schema failures.
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

/// Append adopter categories to an IPC file/stream and return newly encoded file bytes.
///
/// # Errors
/// Propagates IPC decoding/encoding errors and the [`categorize_adopters`] contract.
pub fn categorize_adopters_ipc(data: &[u8], rate_column: &str) -> PolarsResult<Vec<u8>> {
    let frame = read_ipc(data)?;
    let mut output = categorize_adopters(frame, rate_column)?;
    write_ipc_file(&mut output)
}

/// Execute a scenario grid with one shared time column from Arrow IPC file or stream bytes.
///
/// Each `payload.scenarios` entry must contain a simulation object. The validated
/// Arrow `time` column is injected into every simulation's inputs; a conflicting
/// `time` or `t` value in an individual scenario is rejected.
///
/// # Errors
/// IPC/schema failures return a Polars error. Native execution failures remain
/// in the returned JSON envelope and must be checked before consuming results.
pub fn scenario_grid_ipc(data: &[u8], mut payload: Value) -> PolarsResult<String> {
    let frame = read_ipc(data)?;
    let time = inputs(&frame, &[])?;
    let Some(scenarios) = payload.get_mut("scenarios").and_then(Value::as_array_mut) else {
        polars_bail!(SchemaMismatch: "scenario grid requires a scenarios array");
    };
    for scenario in scenarios {
        let Some(simulation) = scenario
            .get_mut("simulation")
            .and_then(Value::as_object_mut)
        else {
            polars_bail!(SchemaMismatch: "scenario simulation must be an object");
        };
        let inputs = simulation.entry("inputs").or_insert_with(|| json!({}));
        let Some(inputs) = inputs.as_object_mut() else {
            polars_bail!(SchemaMismatch: "scenario inputs must be an object");
        };
        if inputs.contains_key("time") || inputs.contains_key("t") {
            polars_bail!(SchemaMismatch: "scenario time must come from the Arrow time column");
        }
        inputs.insert("time".into(), time["time"].clone());
    }
    Ok(innovate_core::execute_json(
        &json!({"schema_version":innovate_core::KERNEL_SCHEMA_VERSION,
            "operation":"execute_scenario_grid","payload":payload})
        .to_string(),
    ))
}

/// Encode the ordered execution rows of a successful grid envelope as Arrow IPC.
/// Nested results, artifacts and errors retain their full JSON representation.
/// Rows must explicitly contain either an object result and null error, or a
/// null result and object error. Paused executions are not marked successful.
///
/// Output columns are `scenario_id`, `model_key`, `successful` and `execution_json`.
/// The boolean reflects absence of a row error and absence of paused status; it
/// is not independent scientific or completeness validation of the nested result.
///
/// # Errors
/// Rejects a failed/non-grid envelope, missing execution rows or missing row IDs/model
/// keys, and propagates Polars/IPC construction failures.
pub fn scenario_executions_ipc(response: &Value) -> PolarsResult<Vec<u8>> {
    let rows = scenario_rows(response, "executions")?;
    let mut ids = Vec::with_capacity(rows.len());
    let mut models = Vec::with_capacity(rows.len());
    let mut successful = Vec::with_capacity(rows.len());
    let mut encoded = Vec::with_capacity(rows.len());
    for row in rows {
        ids.push(
            row["id"]
                .as_str()
                .ok_or_else(|| polars_err!(SchemaMismatch: "missing scenario ID"))?,
        );
        models.push(
            row["model_key"]
                .as_str()
                .ok_or_else(|| polars_err!(SchemaMismatch: "missing scenario model key"))?,
        );
        successful.push(scenario_outcome_has_result(row)? && row["result"]["status"] != "paused");
        encoded.push(row.to_string());
    }
    let mut frame = df!("scenario_id" => ids, "model_key" => models,
        "successful" => successful, "execution_json" => encoded)?;
    write_ipc_file(&mut frame)
}

/// Encode ordered pair and ensemble comparisons, preserving nested outcomes.
/// Each row must explicitly contain an object result and null error, or a null
/// result and object error; incomplete or contradictory outcomes are rejected.
///
/// Returns IPC columns `comparison_index`, `kind`, `successful` and `comparison_json`.
/// The success flag reflects the row's error field; nested outcomes remain JSON
/// and are not independently scientifically validated by this projection.
///
/// # Errors
/// Rejects a failed/non-grid envelope, absent comparison rows or rows without a
/// recognized pair/ensemble shape, and propagates Polars/IPC construction failures.
pub fn scenario_comparisons_ipc(response: &Value) -> PolarsResult<Vec<u8>> {
    let rows = scenario_rows(response, "comparisons")?;
    let mut kinds = Vec::with_capacity(rows.len());
    let mut successful = Vec::with_capacity(rows.len());
    let mut encoded = Vec::with_capacity(rows.len());
    for row in rows {
        let kind = if row["pairs"].is_array() {
            "ensemble"
        } else if row["baseline"].is_string() && row["alternative"].is_string() {
            "pair"
        } else {
            polars_bail!(SchemaMismatch: "invalid scenario comparison row");
        };
        kinds.push(kind);
        successful.push(scenario_outcome_has_result(row)?);
        encoded.push(row.to_string());
    }
    let indices: Vec<_> = (0..rows.len() as u64).collect();
    let mut frame = df!("comparison_index" => indices, "kind" => kinds,
        "successful" => successful, "comparison_json" => encoded)?;
    write_ipc_file(&mut frame)
}

fn scenario_outcome_has_result(row: &Value) -> PolarsResult<bool> {
    match (row.get("result"), row.get("error")) {
        (Some(Value::Object(_)), Some(Value::Null)) => Ok(true),
        (Some(Value::Null), Some(Value::Object(_))) => Ok(false),
        _ => {
            polars_bail!(SchemaMismatch: "scenario outcome requires an explicit object result and null error, or null result and object error")
        }
    }
}

fn scenario_rows<'a>(response: &'a Value, field: &str) -> PolarsResult<&'a [Value]> {
    if response["operation"] != "execute_scenario_grid"
        || response.get("error") != Some(&Value::Null)
    {
        polars_bail!(SchemaMismatch: "expected a successful scenario grid envelope");
    }
    response["result"][field]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| polars_err!(SchemaMismatch: "missing scenario {field} rows"))
}

fn write_ipc_file(frame: &mut DataFrame) -> PolarsResult<Vec<u8>> {
    let compat = CompatLevel::oldest();
    let schema = frame.schema().to_arrow(compat);
    frame.align_chunks_par();
    let mut bytes = Vec::new();
    let mut writer = FileWriter::new(
        &mut bytes,
        schema.into(),
        None,
        WriteOptions { compression: None },
    );
    writer.start()?;
    for batch in frame.iter_chunks(compat, true) {
        writer.write(&batch, None)?;
    }
    writer.finish()?;
    Ok(bytes)
}

/// Encode validated paired Bass trajectories in strategy, draw and time order.
/// Parameter arrays retain their original names, values, units and order as JSON.
///
/// Each output row identifies a strategy/draw/time point and retains time units,
/// perspective and provenance. See [`innovate_core::validate_paired_outcomes`]
/// for the enforced alignment and finite-value contract.
///
/// # Errors
/// Rejects invalid paired-outcome payloads and propagates Polars/IPC encoding
/// failures. Passing this structural contract does not validate economic outcomes.
pub fn paired_outcomes_ipc(payload: &Value) -> PolarsResult<Vec<u8>> {
    innovate_core::validate_paired_outcomes(payload)
        .map_err(|error| polars_err!(ComputeError: "{error}"))?;

    let mut ids = Vec::new();
    let mut draws = Vec::new();
    let mut time_units = Vec::new();
    let mut perspectives = Vec::new();
    let mut provenance = Vec::new();
    let mut parameters = Vec::new();
    let mut times = Vec::new();
    let mut adoption = Vec::new();
    // The native validator established these field types and trajectory lengths.
    for strategy in payload["strategies"].as_array().unwrap() {
        for draw in strategy["draws"].as_array().unwrap() {
            let parameter_json = draw["parameters"].to_string();
            let time = draw["time"].as_array().unwrap();
            let values = draw["adoption"].as_array().unwrap();
            for (time, value) in time.iter().zip(values) {
                ids.push(strategy["scenario_id"].as_str().unwrap());
                draws.push(draw["draw_id"].as_u64().unwrap());
                time_units.push(strategy["time_unit"].as_str().unwrap());
                perspectives.push(strategy["perspective"].as_str());
                provenance.push(strategy["provenance"].as_str().unwrap());
                parameters.push(parameter_json.clone());
                times.push(time.as_f64().unwrap());
                adoption.push(value.as_f64().unwrap());
            }
        }
    }
    let mut frame = df!(
        "scenario_id" => ids, "draw_id" => draws, "time_unit" => time_units,
        "perspective" => perspectives, "provenance" => provenance,
        "parameters_json" => parameters, "time" => times, "adoption" => adoption
    )?;
    write_ipc_file(&mut frame)
}

/// Explicit column mapping for [`bind_panel`] (R03).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnBinding {
    /// Name of the numeric time column.
    pub time: String,
    /// Name of the product identifier column.
    pub product: String,
    /// Name of the measurement identifier column.
    pub measure: String,
    /// Name of the measurement unit column.
    pub unit: String,
    /// Name of the numeric observation column.
    pub value: String,
    /// Optional name of the numeric denominator column.
    pub denominator: Option<String>,
}

/// One validated observation row extracted by [`bind_panel`].
#[derive(Debug, Clone, PartialEq)]
pub struct PanelRow {
    /// Finite observation time.
    pub time: f64,
    /// Product identifier.
    pub product: String,
    /// Measurement identifier.
    pub measure: String,
    /// Declared unit for the product and measurement series.
    pub unit: String,
    /// Observed value, or `None` for a missing observation.
    pub value: Option<f64>,
    /// Optional positive denominator when supplied and present.
    pub denominator: Option<f64>,
}

/// Normalized observation panel produced by [`bind_panel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationPanel {
    /// Validated rows in input order.
    pub rows: Vec<PanelRow>,
}

/// Bind a Polars `DataFrame` into an [`ObservationPanel`] using explicit column names (R03).
///
/// Preserves `null` observations as `None` distinct from `Some(0.0)`, is invariant to
/// column permutation, and rejects missing columns, wrong types, duplicate `(time, product, measure)`
/// keys, non-finite values, or ambiguous units within a `(product, measure)` series.
///
/// # Errors
/// Returns `PolarsError` on schema mismatches, duplicate keys, ambiguous units, or non-finite values.
pub fn bind_panel(frame: DataFrame, binding: &ColumnBinding) -> PolarsResult<ObservationPanel> {
    let n = frame.height();
    let time_col = frame.column(&binding.time)?;
    if !time_col.dtype().is_numeric() {
        polars_bail!(SchemaMismatch: "time column must be numeric");
    }
    let time_cast = time_col.cast(&DataType::Float64)?;
    let time_arr = time_cast.f64()?;
    if time_arr.null_count() != 0 {
        polars_bail!(ComputeError: "time column must not contain nulls");
    }

    let prod_col = frame.column(&binding.product)?.str()?;
    if prod_col.null_count() != 0 {
        polars_bail!(ComputeError: "product column must not contain nulls");
    }
    let meas_col = frame.column(&binding.measure)?.str()?;
    if meas_col.null_count() != 0 {
        polars_bail!(ComputeError: "measure column must not contain nulls");
    }
    let unit_col = frame.column(&binding.unit)?.str()?;
    if unit_col.null_count() != 0 {
        polars_bail!(ComputeError: "unit column must not contain nulls");
    }

    let val_col = frame.column(&binding.value)?;
    if !val_col.dtype().is_numeric() {
        polars_bail!(SchemaMismatch: "value column must be numeric");
    }
    let val_cast = val_col.cast(&DataType::Float64)?;
    let val_arr = val_cast.f64()?;

    let denom_arr = if let Some(ref denom_name) = binding.denominator {
        let dcol = frame.column(denom_name)?;
        if !dcol.dtype().is_numeric() {
            polars_bail!(SchemaMismatch: "denominator column must be numeric");
        }
        Some(dcol.cast(&DataType::Float64)?)
    } else {
        None
    };
    let denom_f64 = denom_arr.as_ref().map(|c| c.f64()).transpose()?;

    let mut seen_keys = std::collections::HashSet::with_capacity(n);
    let mut series_units = std::collections::HashMap::<(String, String), String>::new();
    let mut rows = Vec::with_capacity(n);

    for i in 0..n {
        let t = time_arr.get(i).unwrap();
        if !t.is_finite() {
            polars_bail!(ComputeError: "time values must be finite");
        }
        let product = prod_col.get(i).unwrap().to_string();
        let measure = meas_col.get(i).unwrap().to_string();
        let unit = unit_col.get(i).unwrap().to_string();
        if product.trim().is_empty() || measure.trim().is_empty() || unit.trim().is_empty() {
            polars_bail!(SchemaMismatch: "product, measure, and unit must be nonempty");
        }
        if !seen_keys.insert((t.to_bits(), product.clone(), measure.clone())) {
            polars_bail!(Duplicate: "duplicate (time, product, measure) key in panel");
        }
        if let Some(prev_unit) =
            series_units.insert((product.clone(), measure.clone()), unit.clone())
            && prev_unit != unit
        {
            polars_bail!(SchemaMismatch: "ambiguous units for ({product}, {measure})");
        }
        let value = val_arr.get(i);
        if let Some(v) = value
            && !v.is_finite()
        {
            polars_bail!(ComputeError: "observation values must be finite when present");
        }
        let denominator = denom_f64.as_ref().and_then(|arr| arr.get(i));
        if let Some(d) = denominator
            && (!d.is_finite() || d <= 0.0)
        {
            polars_bail!(ComputeError: "denominator values must be finite and positive when present");
        }
        rows.push(PanelRow {
            time: t,
            product,
            measure,
            unit,
            value,
            denominator,
        });
    }
    Ok(ObservationPanel { rows })
}

/// One row of a versioned [`JointDrawBundle`] (R04).
#[derive(Debug, Clone, PartialEq)]
pub struct JointDrawRow {
    /// Draw identifier used to preserve alignment across rows.
    pub draw_id: u64,
    /// Identifier of the draw's chain.
    pub chain_id: u64,
    /// Scenario identifier.
    pub scenario_id: String,
    /// Strategy identifier within the scenario.
    pub strategy_id: String,
    /// Product identifier.
    pub product: String,
    /// Region identifier.
    pub region: String,
    /// Time associated with the value.
    pub time: f64,
    /// Draw value, or `None` when the value is missing.
    pub value: Option<f64>,
    /// Declared measurement units.
    pub units: String,
    /// Fit status associated with the draw.
    pub fit_status: String,
    /// Provenance label supplied with the draw.
    pub provenance: String,
}

/// Versioned joint posterior/scenario draw bundle (`specs/joint-draw-v1.json`, R04).
#[derive(Debug, Clone, PartialEq)]
pub struct JointDrawBundle {
    /// Version of the joint-draw interchange schema.
    pub schema_version: String,
    /// Explicit label describing the declared draw dependence.
    pub dependence: String,
    /// Rows forming the joint draw bundle.
    pub rows: Vec<JointDrawRow>,
}

fn validate_joint_draw_bundle(bundle: &JointDrawBundle) -> PolarsResult<()> {
    if bundle.schema_version != "1.0" {
        polars_bail!(SchemaMismatch: "unsupported joint-draw schema_version");
    }
    if bundle.dependence.trim().is_empty() {
        polars_bail!(SchemaMismatch: "joint-draw bundle requires explicit dependence label");
    }
    if bundle.rows.is_empty() {
        polars_bail!(ComputeError: "joint-draw bundle must contain at least one row");
    }
    let mut keys = std::collections::HashSet::with_capacity(bundle.rows.len());
    for row in &bundle.rows {
        if row.units.trim().is_empty() {
            polars_bail!(SchemaMismatch: "joint-draw row requires non-empty units");
        }
        if row.fit_status != "valid" {
            polars_bail!(ComputeError: "joint-draw bundle rejects invalid fit_status");
        }
        if row.provenance.trim().is_empty() || !row.time.is_finite() {
            polars_bail!(ComputeError: "joint-draw row requires finite time and provenance");
        }
        if let Some(v) = row.value
            && !v.is_finite()
        {
            polars_bail!(ComputeError: "joint-draw value must be finite when present");
        }
        let key = (
            row.draw_id,
            row.chain_id,
            row.scenario_id.clone(),
            row.strategy_id.clone(),
            row.product.clone(),
            row.region.clone(),
            row.time.to_bits(),
        );
        if !keys.insert(key) {
            polars_bail!(Duplicate: "duplicate joint-draw key");
        }
    }
    Ok(())
}

/// Encode a [`JointDrawBundle`] to Arrow IPC file bytes (R04).
///
/// # Errors
/// Returns `PolarsError` if validation or Arrow IPC encoding fails.
pub fn encode_joint_draws(bundle: &JointDrawBundle) -> PolarsResult<Vec<u8>> {
    validate_joint_draw_bundle(bundle)?;
    let draw_ids: Vec<u64> = bundle.rows.iter().map(|r| r.draw_id).collect();
    let chain_ids: Vec<u64> = bundle.rows.iter().map(|r| r.chain_id).collect();
    let scenario_ids: Vec<&str> = bundle.rows.iter().map(|r| r.scenario_id.as_str()).collect();
    let strategy_ids: Vec<&str> = bundle.rows.iter().map(|r| r.strategy_id.as_str()).collect();
    let products: Vec<&str> = bundle.rows.iter().map(|r| r.product.as_str()).collect();
    let regions: Vec<&str> = bundle.rows.iter().map(|r| r.region.as_str()).collect();
    let times: Vec<f64> = bundle.rows.iter().map(|r| r.time).collect();
    let values: Vec<Option<f64>> = bundle.rows.iter().map(|r| r.value).collect();
    let units: Vec<&str> = bundle.rows.iter().map(|r| r.units.as_str()).collect();
    let fit_statuses: Vec<&str> = bundle.rows.iter().map(|r| r.fit_status.as_str()).collect();
    let provenances: Vec<&str> = bundle.rows.iter().map(|r| r.provenance.as_str()).collect();
    let dependencies: Vec<&str> = vec![bundle.dependence.as_str(); bundle.rows.len()];
    let versions: Vec<&str> = vec![bundle.schema_version.as_str(); bundle.rows.len()];

    let mut frame = df!(
        "schema_version" => versions,
        "dependence" => dependencies,
        "draw_id" => draw_ids,
        "chain_id" => chain_ids,
        "scenario_id" => scenario_ids,
        "strategy_id" => strategy_ids,
        "product" => products,
        "region" => regions,
        "time" => times,
        "value" => values,
        "units" => units,
        "fit_status" => fit_statuses,
        "provenance" => provenances
    )?;
    write_ipc_file(&mut frame)
}

/// Decode and validate Arrow IPC bytes into a [`JointDrawBundle`] (R04).
///
/// # Errors
/// Returns `PolarsError` if IPC decoding fails or the bundle violates `joint-draw-v1.json`.
pub fn decode_joint_draws(bytes: &[u8]) -> PolarsResult<JointDrawBundle> {
    let frame = read_ipc(bytes)?;
    let n = frame.height();
    if n == 0 {
        polars_bail!(ComputeError: "joint-draw IPC contains no rows");
    }
    let versions = frame.column("schema_version")?.str()?;
    let dependencies = frame.column("dependence")?.str()?;
    let draw_ids = frame.column("draw_id")?.u64()?;
    let chain_ids = frame.column("chain_id")?.u64()?;
    let scenario_ids = frame.column("scenario_id")?.str()?;
    let strategy_ids = frame.column("strategy_id")?.str()?;
    let products = frame.column("product")?.str()?;
    let regions = frame.column("region")?.str()?;
    let times = frame.column("time")?.f64()?;
    let values = frame.column("value")?.f64()?;
    let units = frame.column("units")?.str()?;
    let fit_statuses = frame.column("fit_status")?.str()?;
    let provenances = frame.column("provenance")?.str()?;

    let schema_version = versions
        .get(0)
        .ok_or_else(|| polars_err!(SchemaMismatch: "missing schema_version"))?
        .to_string();
    let dependence = dependencies
        .get(0)
        .ok_or_else(|| polars_err!(SchemaMismatch: "missing dependence"))?
        .to_string();

    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(JointDrawRow {
            draw_id: draw_ids
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null draw_id"))?,
            chain_id: chain_ids
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null chain_id"))?,
            scenario_id: scenario_ids
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null scenario_id"))?
                .to_string(),
            strategy_id: strategy_ids
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null strategy_id"))?
                .to_string(),
            product: products
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null product"))?
                .to_string(),
            region: regions
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null region"))?
                .to_string(),
            time: times
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null time"))?,
            value: values.get(i),
            units: units
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null units"))?
                .to_string(),
            fit_status: fit_statuses
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null fit_status"))?
                .to_string(),
            provenance: provenances
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "null provenance"))?
                .to_string(),
        });
    }
    let bundle = JointDrawBundle {
        schema_version,
        dependence,
        rows,
    };
    validate_joint_draw_bundle(&bundle)?;
    Ok(bundle)
}

/// One row of a semi-Markov cohort occupancy/transition trajectory (SM03).
#[derive(Debug, Clone, PartialEq)]
pub struct CohortOccupancyRow {
    /// Draw identifier for the trajectory.
    pub draw_id: u64,
    /// Scenario identifier.
    pub scenario_id: String,
    /// Cohort identifier.
    pub cohort_id: String,
    /// Subgroup label within the cohort.
    pub subgroup: String,
    /// State occupied by the cohort.
    pub state: String,
    /// Discrete duration bin within the state.
    pub duration_bin: u32,
    /// Time associated with the row.
    pub time: f64,
    /// Nonnegative cohort occupancy at that time.
    pub occupancy: f64,
    /// Nonnegative transition flow associated with the row.
    pub transition_flow: f64,
}

/// Encode cohort occupancy/transition rows to Arrow IPC bytes (SM03).
///
/// # Errors
/// Returns `PolarsError` if `rows` is empty, values are non-finite/negative, or IPC encoding fails.
pub fn cohort_occupancy_ipc(rows: &[CohortOccupancyRow]) -> PolarsResult<Vec<u8>> {
    if rows.is_empty() {
        polars_bail!(ComputeError: "cohort occupancy rows must be nonempty");
    }
    for row in rows {
        if !row.time.is_finite()
            || !row.occupancy.is_finite()
            || row.occupancy < 0.0
            || !row.transition_flow.is_finite()
            || row.transition_flow < 0.0
            || row.scenario_id.trim().is_empty()
            || row.cohort_id.trim().is_empty()
            || row.state.trim().is_empty()
        {
            polars_bail!(ComputeError: "cohort occupancy row contains invalid keys or values");
        }
    }
    let draw_ids: Vec<u64> = rows.iter().map(|r| r.draw_id).collect();
    let scenario_ids: Vec<&str> = rows.iter().map(|r| r.scenario_id.as_str()).collect();
    let cohort_ids: Vec<&str> = rows.iter().map(|r| r.cohort_id.as_str()).collect();
    let subgroups: Vec<&str> = rows.iter().map(|r| r.subgroup.as_str()).collect();
    let states: Vec<&str> = rows.iter().map(|r| r.state.as_str()).collect();
    let duration_bins: Vec<u32> = rows.iter().map(|r| r.duration_bin).collect();
    let times: Vec<f64> = rows.iter().map(|r| r.time).collect();
    let occupancies: Vec<f64> = rows.iter().map(|r| r.occupancy).collect();
    let flows: Vec<f64> = rows.iter().map(|r| r.transition_flow).collect();

    let mut frame = df!(
        "draw_id" => draw_ids,
        "scenario_id" => scenario_ids,
        "cohort_id" => cohort_ids,
        "subgroup" => subgroups,
        "state" => states,
        "duration_bin" => duration_bins,
        "time" => times,
        "occupancy" => occupancies,
        "transition_flow" => flows
    )?;
    write_ipc_file(&mut frame)
}

/// Decode cohort occupancy/transition rows from Arrow IPC bytes (SM03).
///
/// # Errors
/// Returns `PolarsError` if IPC decoding or column extraction fails.
pub fn decode_cohort_occupancy_ipc(bytes: &[u8]) -> PolarsResult<Vec<CohortOccupancyRow>> {
    let frame = read_ipc(bytes)?;
    let n = frame.height();
    let draw_ids = frame.column("draw_id")?.u64()?;
    let scenario_ids = frame.column("scenario_id")?.str()?;
    let cohort_ids = frame.column("cohort_id")?.str()?;
    let subgroups = frame.column("subgroup")?.str()?;
    let states = frame.column("state")?.str()?;
    let duration_bins = frame.column("duration_bin")?.u32()?;
    let times = frame.column("time")?.f64()?;
    let occupancies = frame.column("occupancy")?.f64()?;
    let flows = frame.column("transition_flow")?.f64()?;

    let mut out = Vec::with_capacity(n);
    let null_err = || PolarsError::ComputeError("null value in cohort occupancy column".into());
    for i in 0..n {
        out.push(CohortOccupancyRow {
            draw_id: draw_ids.get(i).ok_or_else(null_err)?,
            scenario_id: scenario_ids.get(i).ok_or_else(null_err)?.to_string(),
            cohort_id: cohort_ids.get(i).ok_or_else(null_err)?.to_string(),
            subgroup: subgroups.get(i).ok_or_else(null_err)?.to_string(),
            state: states.get(i).ok_or_else(null_err)?.to_string(),
            duration_bin: duration_bins.get(i).ok_or_else(null_err)?,
            time: times.get(i).ok_or_else(null_err)?,
            occupancy: occupancies.get(i).ok_or_else(null_err)?,
            transition_flow: flows.get(i).ok_or_else(null_err)?,
        });
    }
    Ok(out)
}
