use innovate_core::{KernelBinding, KernelRequest, json};

#[test]
fn dispatch_is_native_without_a_python_launcher() {
    let request = KernelRequest::predict_model(
        "logistic",
        json!({
            "inputs": {"time": [0.0, 1.0, 2.0]},
            "parameters": {"L": 100.0, "k": 1.0, "x0": 1.0}
        }),
    );
    let result = KernelBinding::new().invoke(&request).unwrap();
    assert_eq!(result.metadata["runtime"], "rust_native");
    assert_eq!(result.result.unwrap()["values"][1], 50.0);
}

#[test]
fn unknown_models_never_fall_back_to_python() {
    let error = KernelBinding::new()
        .predict_model("not_a_model", json!({}))
        .unwrap_err();
    assert_eq!(error.code, "unsupported_native_operation");
}

#[test]
fn dispatch_rejects_unknown_schema_versions() {
    let mut request = KernelRequest::discover_models();
    request.schema_version = "999".into();
    assert_eq!(
        KernelBinding::new().invoke(&request).unwrap_err().code,
        "invalid_request"
    );
}

#[test]
fn discovery_does_not_advertise_unported_models() {
    let models = KernelBinding::new().discover_models().unwrap().models;
    assert_eq!(models.len(), 16);
    assert!(models.iter().all(|m| m.supported_backends == ["rust"]));
}

#[test]
fn empty_model_is_a_structured_error_not_a_panic() {
    let request = KernelRequest::predict_model("", json!({}));
    assert_eq!(
        KernelBinding::new().invoke(&request).unwrap_err().code,
        "invalid_request"
    );
}

#[test]
fn direct_parameter_payloads_work_for_bass_and_reject_ambiguous_state() {
    let binding = KernelBinding::new();
    let parameters = json!({"p":0.03,"q":0.35,"m":100.0});
    let result = binding
        .predict_model(
            "bass",
            json!({"inputs":{"time":[3.0,4.0]},"parameters":parameters}),
        )
        .unwrap();
    assert_eq!(result.metadata["runtime"], "rust_native");
    assert!(
        binding
            .predict_model(
                "bass",
                json!({"inputs":{"time":[3.0,4.0]},"parameters":parameters,
        "state":{"model_key":"bass","parameters":{"p":0.3,"q":0.5,"m":99.0}}})
            )
            .is_err()
    );
}

#[test]
fn undeclared_covariates_never_fall_back_after_all_model_families_are_ported() {
    let error=KernelBinding::new().predict_model("bass",json!({"inputs":{"time":[0.],"covariates":{"price":[1.]}},"parameters":{"p":0.03,"q":0.35,"m":100.}})).unwrap_err();
    assert_eq!(error.code, "unsupported_native_operation");
}

#[test]
fn batch_keeps_independent_fits_errors_and_ragged_forecasts_in_order() {
    use innovate_core::{KernelBinding, json};
    let binding = KernelBinding::new();
    let datasets = [100., 200.].map(|capacity| {
        let time: Vec<_> = (0..20).map(f64::from).collect();
        let observed: Vec<_> = time
            .iter()
            .map(|&t| capacity / (1. + (-(t - 5.)).exp()))
            .collect();
        json!({"inputs":{"time":time,"observed":observed}})
    });
    let fits = binding.fit_batch(
        "logistic",
        [datasets[0].clone(), json!({}), datasets[1].clone()],
    );
    assert!(fits[1].is_err());
    let forecasts = binding.predict_batch(
        "logistic",
        [0, 2].map(|index| {
            let fit = fits[index].as_ref().unwrap().result.as_ref().unwrap();
            let time = if index == 0 { vec![5.] } else { vec![5., 6.] };
            json!({"inputs":{"time":time},"state":fit["state"]})
        }),
    );
    for (i, forecast) in forecasts.into_iter().enumerate() {
        let forecast = forecast.unwrap().result.unwrap();
        let values = forecast["values"].as_array().unwrap();
        assert_eq!(values.len(), i + 1);
        assert!((values[0].as_f64().unwrap() - 50. * (i + 1) as f64).abs() < 1e-6);
    }
}

#[test]
fn batch_json_validates_envelope_and_isolates_bad_datasets() {
    use innovate_core::{execute_json, json};
    let execute = |request: serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&execute_json(&request.to_string())).unwrap()
    };
    let mut request = json!({
        "schema_version":"2.0","operation":"predict_batch","model_key":"logistic",
        "payload":{"datasets":[
            {"inputs":{"time":[5.]},"parameters":{"L":100.,"k":1.,"x0":5.}},
            null,
            {"inputs":{"time":[5.,6.]},"parameters":{"L":200.,"k":1.,"x0":5.}}
        ]}
    });
    let response = execute(request.clone());
    assert!(response["error"].is_null());
    let rows = &response["result"]["datasets"];
    assert_eq!(rows[0]["result"]["values"], json!([50.]));
    assert_eq!(rows[1]["error"]["code"], "invalid_request");
    assert_eq!(rows[2]["result"]["values"].as_array().unwrap().len(), 2);
    request["payload"]["workers"] = json!(2);
    assert_eq!(execute(request.clone()), response);
    request["payload"]["workers"] = json!(33);
    assert_eq!(execute(request.clone())["error"]["code"], "invalid_request");
    request["payload"]["workers"] = json!(2);
    request["schema_version"] = json!("1.0");
    assert_eq!(execute(request.clone())["error"]["code"], "invalid_request");
    request["schema_version"] = json!("2.0");
    request["payload"]["datasets"] = json!({});
    assert_eq!(execute(request)["error"]["code"], "invalid_request");
}

#[test]
fn plain_diffusion_rates_match_analytic_reference_points() {
    use innovate_core::{KernelBinding, json};
    for (key, parameters, time, expected) in [
        ("bass", json!({"p":0.1,"q":0.3,"m":100.}), 0., 10.),
        (
            "gompertz",
            json!({"a":100.,"b":1.,"c":0.5}),
            0.,
            50. / std::f64::consts::E,
        ),
        ("fisher_pry", json!({"alpha":2.,"t0":5.}), 5., 0.5),
    ] {
        let r = KernelBinding::new()
            .predict_model(
                key,
                json!({"inputs":{"time":[time]},"parameters":parameters}),
            )
            .unwrap()
            .result
            .unwrap();
        assert!((r["rates"][0][0].as_f64().unwrap() - expected).abs() < 1e-10);
    }
}
