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
        {"id":"b","model_key":"unknown","result":null,"error":{"code":"unsupported_native_operation"}}
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
        frame.column("successful").unwrap().bool().unwrap().get(1),
        Some(false)
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
