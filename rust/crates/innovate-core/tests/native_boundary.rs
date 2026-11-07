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

#[test]
fn bass_rate_remains_finite_when_imitation_innovation_ratio_overflows() {
    let p = 1e-200_f64;
    let q = 1e200_f64;
    let elapsed = q.ln() - p.ln();
    let result = innovate_core::KernelBinding::new()
        .predict_model(
            "bass",
            innovate_core::json!({
                "inputs":{"time":[0.,elapsed/q]},"parameters":{"p":p,"q":q,"m":1.}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(result["rates"][0][0], p);
    assert!((result["values"][1].as_f64().unwrap() - 0.5).abs() < 1e-11);
    let midpoint_rate = result["rates"][1][0].as_f64().unwrap();
    assert!((midpoint_rate / (q / 4.) - 1.).abs() < 1e-11);
}

#[test]
fn rolling_average_recovers_small_terms_after_large_value_exits() {
    let request = innovate_core::json!({
        "schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"rolling_average","window":2,"values":[1e100,1.,3.]}
    });
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        result["result"]["values"],
        innovate_core::json!([null, 5e99, 2.])
    );
}

#[test]
fn calendar_aggregation_fills_days_and_months_but_skips_empty_weeks() {
    let aggregate = |dates, frequency| {
        let request = innovate_core::json!({"schema_version":"2.0","operation":"aggregate_dates",
            "payload":{"dates":dates,"values":[3,2],"frequency":frequency}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let daily = aggregate(["2024-01-03", "2024-01-01"], "D");
    assert_eq!(
        daily["result"]["values"],
        innovate_core::json!([2., 0., 3.])
    );
    let weekly = aggregate(["2024-01-23", "2024-01-02"], "W");
    assert_eq!(
        weekly["result"]["dates"],
        innovate_core::json!(["2024-01-02", "2024-01-23"])
    );
    let monthly = aggregate(["2024-03-31", "2024-01-02"], "M");
    assert_eq!(monthly["result"]["dates"][1], "2024-02-29");
    assert_eq!(monthly["result"]["values"][1], 0.);
    assert_eq!(
        aggregate(["2024-02-30", "2024-01-02"], "M")["error"]["code"],
        "invalid_request"
    );
}

#[test]
fn loess_trend_recovers_linear_values_at_interior_and_edges() {
    let request = innovate_core::json!({
        "schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"loess_trend","window":5,"values":[1.,3.,5.,7.,9.,11.,13.]}
    });
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    for (index, estimate) in result["result"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!((estimate.as_f64().unwrap() - (1. + 2. * index as f64)).abs() < 1e-12);
    }
}

#[test]
fn seasonal_decomposition_recovers_linear_trend_and_periodic_signal() {
    let pattern = [-3., 2., 1., 0.];
    let values: Vec<_> = (0..32)
        .map(|i| 10. + i as f64 * 0.5 + pattern[i % 4])
        .collect();
    let request = innovate_core::json!({
        "schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"seasonal_decomposition","period":4,"window":9,"values":values}
    });
    let envelope: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &envelope["result"];
    for i in 0..values.len() {
        assert!((result["trend"][i].as_f64().unwrap() - (10. + i as f64 * 0.5)).abs() < 1e-10);
        assert!((result["seasonal"][i].as_f64().unwrap() - pattern[i % 4]).abs() < 1e-10);
        assert!(result["residual"][i].as_f64().unwrap().abs() < 1e-10);
    }
}

#[test]
fn robust_seasonal_decomposition_keeps_an_outlier_in_the_residual() {
    let pattern = [-3., 2., 1., 0.];
    let mut values: Vec<_> = (0..80)
        .map(|i| 10. + i as f64 * 0.5 + pattern[i % 4])
        .collect();
    values[40] += 100.;
    let request = innovate_core::json!({
        "schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"seasonal_decomposition","period":4,"window":13,"values":values,"robust":true}
    });
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!((response["result"]["residual"][40].as_f64().unwrap() - 100.).abs() < 0.1);
}

#[test]
fn nonrobust_seasonal_matches_independent_stl_reference() {
    assert_stl_reference(false, "nonrobust", 1e-10);
}

#[test]
fn historical_robust_seasonal_matches_independent_stl_reference() {
    assert_stl_reference(true, "robust", 1e-9);
}

fn assert_stl_reference(robust: bool, reference: &str, tolerance: f64) {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/stl-reference-20261107/reference.json"
    )).unwrap();
    let request = innovate_core::json!({"schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"seasonal_decomposition","period":4,"window":13,"values":fixture["values"],"robust":robust,"historical":robust}});
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    for (key, reference_key) in [
        ("trend", "trend"),
        ("seasonal", "seasonal"),
        ("residual", "remainder"),
    ] {
        let actual = result["result"][key].as_array().unwrap();
        let expected = fixture["reference"][reference][reference_key]
            .as_array()
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < tolerance);
        }
    }
}

#[test]
fn stationary_sarima_filter_matches_independent_one_step_forecasts() {
    let mut fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/sarima-reference/stationary-fixed.json"
    )).unwrap();
    fixture["method"] = "sarima_filter".into();
    let request = innovate_core::json!({
        "schema_version": "2.0", "operation": "preprocess_series", "payload": fixture
    });
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    for field in ["fitted", "residual"] {
        let actual = result["result"][field].as_array().unwrap();
        let expected = fixture[field].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            let error = (a.as_f64().unwrap() - b.as_f64().unwrap()).abs();
            assert!(error < 1e-10, "{field}[{i}] error {error}");
        }
    }
}

#[test]
fn sarima_combined_differencing_preserves_observation_alignment() {
    // A linear trend plus a repeating seasonal effect has zero combined
    // differences except at the isolated shock and its lagged echoes.
    let request = innovate_core::json!({
        "schema_version":"2.0", "operation":"preprocess_series",
        "payload":{"method":"sarima_filter","order":[0,1,0],"seasonal_order":[0,1,0,3],
                   "coefficients":[],"values":[2,5,3,5,8,6,8,21,9,11,14,12]}
    });
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        response["result"]["residual"],
        innovate_core::json!([null, null, null, null, 0., 0., 0., 10., -10., 0., -10., 10.])
    );
    assert_eq!(
        response["result"]["fitted"],
        innovate_core::json!([null, null, null, null, 8., 6., 8., 11., 19., 11., 24., 2.])
    );
}
