use innovate_tabular::predict;
use polars_arrow::datatypes::Metadata;
use polars_arrow::io::ipc::read::{
    FileReader, StreamReader, StreamState, read_file_metadata, read_stream_metadata,
};
use polars_arrow::io::ipc::write::{FileWriter, StreamWriter, WriteOptions};
use polars_core::prelude::*;
use serde_json::json;
use std::io::Cursor;

fn write_ipc_file_with_metadata(
    frame: &mut DataFrame,
    metadata: Option<std::sync::Arc<Metadata>>,
) -> Vec<u8> {
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
    if let Some(metadata) = metadata {
        writer.set_custom_schema_metadata(metadata);
    }
    writer.start().unwrap();
    for batch in frame.iter_chunks(compat, true) {
        writer.write(&batch, None).unwrap();
    }
    writer.finish().unwrap();
    bytes
}

fn write_ipc_stream_with_metadata(
    frame: &mut DataFrame,
    metadata: Option<std::sync::Arc<Metadata>>,
) -> Vec<u8> {
    let compat = CompatLevel::oldest();
    let schema = frame.schema().to_arrow(compat);
    frame.align_chunks_par();
    let mut bytes = Vec::new();
    let mut writer = StreamWriter::new(&mut bytes, WriteOptions { compression: None });
    if let Some(metadata) = metadata {
        writer.set_custom_schema_metadata(metadata);
    }
    writer.start(&schema, None).unwrap();
    for batch in frame.iter_chunks(compat, true) {
        writer.write(&batch, None).unwrap();
    }
    writer.finish().unwrap();
    bytes
}

fn read_ipc_file_with_metadata(
    bytes: &[u8],
) -> (DataFrame, ArrowSchema, Option<std::sync::Arc<Metadata>>) {
    let mut cursor = Cursor::new(bytes);
    let metadata = read_file_metadata(&mut cursor).unwrap();
    let schema = metadata.schema.as_ref().clone();
    let custom = metadata.custom_schema_metadata.clone();
    let polars_schema = Schema::from_arrow_schema(&schema);
    let mut frame = DataFrame::empty_with_schema(&polars_schema);
    for batch in FileReader::new(cursor, metadata, None, None) {
        frame.append_record_batch(batch.unwrap()).unwrap();
    }
    (frame, schema, custom)
}

fn read_ipc_stream_with_metadata(
    bytes: &[u8],
) -> (DataFrame, ArrowSchema, Option<std::sync::Arc<Metadata>>) {
    let mut cursor = Cursor::new(bytes);
    let metadata = read_stream_metadata(&mut cursor).unwrap();
    let schema = metadata.schema.clone();
    let custom = metadata
        .custom_schema_metadata
        .clone()
        .map(std::sync::Arc::new);
    let polars_schema = Schema::from_arrow_schema(&schema);
    let mut reader = StreamReader::new(cursor, metadata, None);
    let mut frame = DataFrame::empty_with_schema(&polars_schema);
    for state in &mut reader {
        if let StreamState::Some(batch) = state.unwrap() {
            frame.append_record_batch(batch).unwrap();
        }
    }
    (frame, schema, custom)
}

#[test]
fn prediction_preserves_identity_and_adds_native_values() {
    let frame = df!("id" => ["a", "b", "c"], "time" => [0.0, 1.0, 2.0]).unwrap();
    let out = predict(frame, "logistic", json!({"L":100.0,"k":1.0,"x0":1.0})).unwrap();
    assert_eq!(out.column("id").unwrap().str().unwrap().get(1), Some("b"));
    assert_eq!(
        out.column("prediction").unwrap().f64().unwrap().get(1),
        Some(50.0)
    );
}

#[test]
fn invalid_time_contract_fails_in_rust() {
    for times in [vec![0.0, f64::NAN], vec![1.0, 0.0], vec![0.0, 0.0]] {
        let frame = df!("time" => times).unwrap();
        assert!(predict(frame, "logistic", json!({"L":100.0,"k":1.0,"x0":1.0})).is_err());
    }
}

#[test]
fn multivariate_predictions_and_signed_rates_preserve_all_columns() {
    let frame = df!("id"=>[10,20,30],"time"=>[0.,10.,20.]).unwrap();
    let params = json!({"p1":0.03,"q1":0.3,"m1":100.,"p2":0.04,"q2":0.4,"m2":200.});
    let options =
        json!({"model_options":{"variant":"successive_generations","launch_times":[0.,5.]}});
    let out = innovate_tabular::predict_with_options(
        frame.clone(),
        "norton_bass",
        params.clone(),
        options.clone(),
    )
    .unwrap();
    assert_eq!(out.width(), 4);
    assert_eq!(out.column("id").unwrap(), frame.column("id").unwrap());
    assert!(
        innovate_tabular::predict_with_options(out, "norton_bass", params.clone(), options.clone())
            .is_err()
    );
    let mut options = options;
    options["quantity"] = json!("rate");
    let rates =
        innovate_tabular::predict_with_options(frame, "norton_bass", params, options).unwrap();
    assert!(
        rates
            .column("rate_1")
            .unwrap()
            .f64()
            .unwrap()
            .get(2)
            .unwrap()
            < 0.
    );
}
#[test]
fn invalid_quantities_and_input_types_fail_without_silent_defaults() {
    for options in [
        json!({"quantity":false}),
        json!({"quantity":"incidence"}),
        json!({"unknown":true}),
        json!([]),
    ] {
        assert!(
            innovate_tabular::predict_with_options(
                df!("time"=>[0.,1.]).unwrap(),
                "logistic",
                json!({"L":1.,"k":1.,"x0":1.}),
                options
            )
            .is_err()
        );
    }
    for frame in [
        df!("time"=>[Some(0.),None]).unwrap(),
        df!("time"=>["a","b"]).unwrap(),
        df!("time"=>Vec::<f64>::new()).unwrap(),
    ] {
        assert!(predict(frame, "logistic", json!({"L":1.,"k":1.,"x0":1.})).is_err());
    }
}

#[test]
fn adopter_categories_use_historical_inclusive_boundaries() {
    let frame = df!(
        "time" => [-2., -1., 0., 1., 2.],
        "adoption_rate" => [0., 1e308, 0., 1e308, 0.],
        "id" => ["a", "b", "c", "d", "e"],
    )
    .unwrap();
    let tiny = f64::from_bits(1);
    let extreme = df!(
        "time" => [-f64::MAX, 0., tiny, 2. * tiny, f64::MAX],
        "adoption_rate" => [0., 1., 0., 1., 0.],
        "id" => ["a", "b", "c", "d", "e"],
    )
    .unwrap();
    for frame in [frame, extreme] {
        let result = innovate_tabular::categorize_adopters(frame.clone(), "adoption_rate").unwrap();
        assert_eq!(
            result
                .column("category")
                .unwrap()
                .str()
                .unwrap()
                .iter()
                .map(Option::unwrap)
                .collect::<Vec<_>>(),
            [
                "Innovators",
                "Early Adopters",
                "Early Majority",
                "Late Majority",
                "Laggards"
            ]
        );
        for name in ["id", "time", "adoption_rate"] {
            assert_eq!(result.column(name).unwrap(), frame.column(name).unwrap());
        }
    }
    let negative = df!("time" => [0., 1.], "adoption_rate" => [-1., 2.]).unwrap();
    assert!(innovate_tabular::categorize_adopters(negative, "adoption_rate").is_err());
}

#[test]
fn arrow_scenario_grid_supplies_time_and_rejects_conflicting_times() {
    let mut frame = df!("time" => [0., 1.]).unwrap();
    let bytes = write_ipc_file_with_metadata(&mut frame, None);
    let mut payload = json!({"scenarios":[{"id":"base","model_key":"logistic",
        "simulation":{"parameters":{"L":100.,"k":1.,"x0":1.}}}]});
    let response = innovate_tabular::scenario_grid_ipc(&bytes, payload.clone()).unwrap();
    let stream = write_ipc_stream_with_metadata(&mut frame, None);
    assert_eq!(
        innovate_tabular::scenario_grid_ipc(&stream, payload.clone()).unwrap(),
        response
    );
    assert!(innovate_tabular::scenario_grid_ipc(b"invalid IPC", payload.clone()).is_err());
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(
        response["result"]["executions"][0]["result"]["values"][1],
        50.
    );
    payload["scenarios"][0]["simulation"]["inputs"] = json!({"t":[2.]});
    assert!(innovate_tabular::scenario_grid_ipc(&bytes, payload).is_err());
    assert!(innovate_tabular::scenario_grid_ipc(&bytes, json!({"scenarios":[1]})).is_err());
}

#[test]
fn scenario_arrow_execution_rows_preserve_nested_data_and_order() {
    let rows = json!([
        {"id":"a","model_key":"bass","result":{"counts":[1,2],"metadata":{"seed":u64::MAX}},"error":null},
        {"id":"b","model_key":"unknown","result":null,"error":{"code":"unsupported_native_operation"}},
        {"id":"c","model_key":"bass","result":{"status":"paused","checkpoint":"opaque"},"error":null}
    ]);
    let envelope = json!({"operation":"execute_scenario_grid","error":null,
        "result":{"executions":rows}});
    let bytes = innovate_tabular::scenario_executions_ipc(&envelope).unwrap();
    let (frame, _, _) = read_ipc_file_with_metadata(&bytes);
    assert_eq!(
        frame.column("scenario_id").unwrap().str().unwrap().get(0),
        Some("a")
    );
    assert_eq!(
        frame
            .column("successful")
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some(true), Some(false), Some(false)]
    );
    for (index, row) in rows.as_array().unwrap().iter().enumerate() {
        let encoded = frame
            .column("execution_json")
            .unwrap()
            .str()
            .unwrap()
            .get(index)
            .unwrap();
        assert_eq!(
            &serde_json::from_str::<serde_json::Value>(encoded).unwrap(),
            row
        );
    }
}

#[test]
fn scenario_arrow_comparisons_preserve_pair_and_ensemble_rows() {
    let rows = json!([
        {"baseline":"a","alternative":"b","result":{"difference":[2.5]},"error":null},
        {"pairs":[{"baseline":"a","alternative":"b"}],"result":null,"error":{"code":"invalid_comparison"}}
    ]);
    let response = json!({"operation":"execute_scenario_grid","error":null,
        "result":{"comparisons":rows}});
    let bytes = innovate_tabular::scenario_comparisons_ipc(&response).unwrap();
    let (frame, _, _) = read_ipc_file_with_metadata(&bytes);
    assert_eq!(
        frame.column("kind").unwrap().str().unwrap().get(1),
        Some("ensemble")
    );
    assert_eq!(
        frame.column("successful").unwrap().bool().unwrap().get(1),
        Some(false)
    );
    let encoded = frame.column("comparison_json").unwrap().str().unwrap();
    for (index, row) in rows.as_array().unwrap().iter().enumerate() {
        assert_eq!(
            &serde_json::from_str::<serde_json::Value>(encoded.get(index).unwrap()).unwrap(),
            row
        );
    }
}

#[test]
fn export_ipc_file_frames_schema_and_preserves_nulls_and_types() {
    let mut frame =
        df!("n" => [1_i32, 2], "value" => [Some(2.), None], "name" => ["a", "β"]).unwrap();
    let metadata = std::sync::Arc::new(
        [("innovate.provenance".into(), "synthetic β".into())]
            .into_iter()
            .collect(),
    );
    let stream = write_ipc_stream_with_metadata(&mut frame, Some(std::sync::Arc::clone(&metadata)));
    let source_file =
        write_ipc_file_with_metadata(&mut frame, Some(std::sync::Arc::clone(&metadata)));

    for input in [stream, source_file] {
        let bytes = innovate_tabular::export_ipc_file(&input).unwrap();
        assert_eq!(&bytes[..12], b"ARROW1\0\0\xff\xff\xff\xff");
        let (decoded, _, custom) = read_ipc_file_with_metadata(&bytes);
        assert_eq!(custom.as_ref(), Some(&metadata));
        assert_eq!(decoded, frame);

        let bytes = innovate_tabular::export_ipc_stream(&input).unwrap();
        let (decoded, _, custom) = read_ipc_stream_with_metadata(&bytes);
        assert_eq!(custom.as_ref(), Some(&metadata));
        assert_eq!(decoded, frame);
    }
}

#[test]
fn ipc_rewrite_keeps_annotations_for_compatible_field_types() {
    use polars_arrow::io::ipc::write::{StreamWriter, WriteOptions};

    use polars_arrow::datatypes::ArrowDataType::LargeList;

    let values = Series::new("".into(), [Some(1_i32), None]);
    let frame = df!("duration" => [Some(1_i32), None],
        "samples" => Series::new("samples".into(), [Some(values), None]),
        "label" => [Some("β"), None],
        "payload" => [Some(&[0, 255, 128][..]), None])
    .unwrap();
    let compat = CompatLevel::newest();
    let mut schema = frame.schema().to_arrow(compat);
    let field = schema.get_mut("duration").unwrap();
    let metadata = || [("units".into(), "days".into())].into_iter().collect();
    *field = field.clone().with_metadata(metadata());
    let label = schema.get_mut("label").unwrap();
    *label = label.clone().with_metadata(metadata());
    let payload = schema.get_mut("payload").unwrap();
    assert_eq!(
        payload.dtype,
        polars_arrow::datatypes::ArrowDataType::BinaryView
    );
    *payload = payload.clone().with_metadata(metadata());
    let LargeList(item) = &mut schema.get_mut("samples").unwrap().dtype else {
        panic!("expected a large list");
    };
    **item = item.clone().with_metadata(metadata());
    let assert_annotations = |schema: &ArrowSchema| {
        let label = schema.get("label").unwrap();
        assert_eq!(
            label.dtype,
            polars_arrow::datatypes::ArrowDataType::LargeUtf8
        );
        assert_eq!(label.metadata.as_deref(), Some(&metadata()));
        let payload = schema.get("payload").unwrap();
        assert_eq!(
            payload.dtype,
            polars_arrow::datatypes::ArrowDataType::LargeBinary
        );
        assert_eq!(payload.metadata.as_deref(), Some(&metadata()));
        assert_eq!(
            schema.get("duration").unwrap().metadata.as_deref(),
            Some(&metadata())
        );
        let LargeList(item) = &schema.get("samples").unwrap().dtype else {
            panic!("expected a large list");
        };
        assert_eq!(item.metadata.as_deref(), Some(&metadata()));
    };
    let mut source = Vec::new();
    let mut writer = StreamWriter::new(&mut source, WriteOptions { compression: None });
    writer.start(&schema, None).unwrap();
    for batch in frame.iter_chunks(compat, true) {
        writer.write(&batch, None).unwrap();
    }
    writer.finish().unwrap();

    let file = innovate_tabular::export_ipc_file(&source).unwrap();
    let (_, file_schema, _) = read_ipc_file_with_metadata(&file);
    assert_annotations(&file_schema);
    for input in [source, file] {
        let stream = innovate_tabular::export_ipc_stream(&input).unwrap();
        let (decoded, stream_schema, _) = read_ipc_stream_with_metadata(&stream);
        assert_annotations(&stream_schema);
        assert_eq!(decoded, frame);
    }
}

#[test]
fn paired_ipc_preserves_order_and_rejects_reordered_draws() {
    use innovate_tabular::paired_outcomes_ipc;

    let parameters = json!([
        {"name":"m","value":100,"unit":"people"},
        {"name":"p","value":0.03,"unit":"year^-1"},
        {"name":"q","value":0.3,"unit":"year^-1"}
    ]);
    let strategy = |id, value| {
        json!({
            "scenario_id":id,"time_unit":"year","provenance":"synthetic",
            "draws":[
                {"draw_id":u64::MAX,"time":[0,1],"adoption":[0,value],"parameters":parameters},
                {"draw_id":2,"time":[0,1],"adoption":[0,2],"parameters":parameters}
            ]
        })
    };
    let mut payload = json!({"strategies":[strategy("baseline",3),strategy("alternative",4)]});
    let bytes = paired_outcomes_ipc(&payload).unwrap();
    let (frame, _, _) = read_ipc_file_with_metadata(&bytes);
    assert_eq!(frame.height(), 8);
    let draws = frame.column("draw_id").unwrap().u64().unwrap();
    assert!(
        draws
            .into_no_null_iter()
            .eq([u64::MAX, u64::MAX, 2, 2, u64::MAX, u64::MAX, 2, 2])
    );
    assert_eq!(
        frame.column("adoption").unwrap().f64().unwrap().get(5),
        Some(4.0)
    );
    assert_eq!(
        frame.column("scenario_id").unwrap().str().unwrap().get(4),
        Some("alternative")
    );
    assert_eq!(frame.column("perspective").unwrap().null_count(), 8);
    let encoded = frame
        .column("parameters_json")
        .unwrap()
        .str()
        .unwrap()
        .get(5)
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(encoded).unwrap(),
        parameters
    );
    let mut positional = payload.clone();
    let draw = &mut positional["strategies"][0]["draws"][0];
    *draw = json!([
        draw["draw_id"],
        draw["parameters"],
        draw["time"],
        draw["adoption"]
    ]);
    assert!(
        paired_outcomes_ipc(&positional)
            .unwrap_err()
            .to_string()
            .contains("must be objects")
    );
    let mut negative = payload.clone();
    negative["strategies"][1]["draws"][0]["adoption"][1] = json!(-0.01);
    assert!(
        paired_outcomes_ipc(&negative)
            .unwrap_err()
            .to_string()
            .contains("paired Bass adoption values must be nonnegative")
    );
    payload["strategies"][1]["draws"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    assert!(
        paired_outcomes_ipc(&payload)
            .unwrap_err()
            .to_string()
            .contains("matching ordered draw IDs")
    );
}
