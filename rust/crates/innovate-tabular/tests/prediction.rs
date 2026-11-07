use innovate_tabular::predict;
use polars::prelude::*;
use serde_json::json;

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
    assert_eq!(result.column("id").unwrap(), frame.column("id").unwrap());
    let negative = df!("time" => [0., 1.], "adoption_rate" => [-1., 2.]).unwrap();
    assert!(innovate_tabular::categorize_adopters(negative, "adoption_rate").is_err());
}

#[test]
fn arrow_scenario_grid_supplies_time_and_rejects_conflicting_times() {
    let mut frame = df!("time" => [0., 1.]).unwrap();
    let mut bytes = Vec::new();
    IpcWriter::new(&mut bytes).finish(&mut frame).unwrap();
    let mut payload = json!({"scenarios":[{"id":"base","model_key":"logistic",
        "simulation":{"parameters":{"L":100.,"k":1.,"x0":1.}}}]});
    let response = innovate_tabular::scenario_grid_ipc(&bytes, payload.clone()).unwrap();
    let mut stream = Vec::new();
    IpcStreamWriter::new(&mut stream)
        .finish(&mut frame)
        .unwrap();
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
    let frame = IpcReader::new(std::io::Cursor::new(bytes))
        .finish()
        .unwrap();
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
    let frame = IpcReader::new(std::io::Cursor::new(bytes))
        .finish()
        .unwrap();
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
    let mut stream = Vec::new();
    let mut stream_writer = IpcStreamWriter::new(&mut stream);
    stream_writer.set_custom_schema_metadata(std::sync::Arc::clone(&metadata));
    stream_writer.finish(&mut frame).unwrap();

    let mut source_file = Vec::new();
    let mut file_writer = IpcWriter::new(&mut source_file);
    file_writer.set_custom_schema_metadata(std::sync::Arc::clone(&metadata));
    file_writer.finish(&mut frame).unwrap();

    for input in [stream, source_file] {
        let bytes = innovate_tabular::export_ipc_file(&input).unwrap();
        assert_eq!(&bytes[..12], b"ARROW1\0\0\xff\xff\xff\xff");
        let mut reader = IpcReader::new(std::io::Cursor::new(bytes));
        assert_eq!(reader.custom_metadata().unwrap().as_ref(), Some(&metadata));
        assert_eq!(reader.finish().unwrap(), frame);

        let bytes = innovate_tabular::export_ipc_stream(&input).unwrap();
        let mut reader = IpcStreamReader::new(std::io::Cursor::new(bytes));
        assert_eq!(reader.custom_metadata().unwrap().as_ref(), Some(&metadata));
        assert_eq!(reader.finish().unwrap(), frame);
    }
}

#[test]
fn paired_ipc_preserves_order_and_rejects_reordered_draws() {
    use innovate_tabular::paired_outcomes_ipc;
    use std::io::Cursor;

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
    let frame = IpcReader::new(Cursor::new(bytes)).finish().unwrap();
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
