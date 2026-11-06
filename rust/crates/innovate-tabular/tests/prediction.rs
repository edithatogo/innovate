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
        json!({"quantity":"rate"}),
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
