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
    let likelihood = result["result"]["log_likelihood"].as_f64().unwrap();
    let expected = fixture["log_likelihood"].as_f64().unwrap();
    assert!((likelihood - expected).abs() < 1e-10);
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

#[test]
fn sarima_fit_matches_independent_ar1_maximum_likelihood() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/sarima-reference/stationary-fixed.json"
    )).unwrap();
    let request = innovate_core::json!({
        "schema_version":"2.0", "operation":"preprocess_series",
        "payload":{"method":"sarima_fit","order":[1,0,0],"seasonal_order":[0,0,0,0],
                   "values":fixture["values"]}
    });
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    // R stats::arima, method = "ML", include.mean = FALSE.
    assert!((result["coefficients"][0].as_f64().unwrap() - 0.8077160397020451).abs() < 2e-6);
    assert!((result["log_likelihood"].as_f64().unwrap() + 27.279117220748503).abs() < 1e-9);
}

#[test]
fn historical_sarima_retains_sarimax_initial_predictions() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../conductor/tracks/scientific_ecosystem_alignment_20261029/evidence/sarima-reference/historical-initialization.json"
    )).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let request = innovate_core::json!({
            "schema_version":"2.0", "operation":"preprocess_series",
            "payload":{"method":"sarima_filter","historical":true,"values":fixture["values"],
                       "order":case["order"],"seasonal_order":case["seasonal_order"],
                       "coefficients":case["coefficients"]}
        });
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        let fitted = response["result"]["fitted"].as_array().unwrap();
        let expected = case["fitted"].as_array().unwrap();
        assert_eq!(fitted.len(), expected.len());
        for (actual, expected) in fitted.iter().zip(expected) {
            assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-8);
        }
    }
}

#[test]
fn offset_timestamp_weekly_buckets_use_elapsed_time() {
    let request = innovate_core::json!({
        "schema_version":"2.0", "operation":"aggregate_dates",
        "payload":{"frequency":"W","dates":[
            "2026-11-08T10:00:00+10:00","2026-11-01T00:00:00Z","2026-11-07T23:59:59.999Z"
        ],"values":[4,1,2]}
    });
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        response["result"],
        innovate_core::json!({
            "dates":["2026-11-01T00:00:00Z","2026-11-08T00:00:00Z"],"values":[3.,4.]
        })
    );
}

#[test]
fn causal_summaries_match_analytic_effects() {
    use innovate_core::json;
    let run = |payload| {
        let request =
            json!({"schema_version":"2.0","operation":"summarize_causal","payload":payload});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let did = run(
        json!({"method":"pre_post","pre_treated":[1,3],"post_treated":[7,9],
                        "pre_control":[2],"post_control":[5]}),
    );
    assert_eq!(did["result"]["diff_in_diff"], 3.);
    let counterfactual =
        run(json!({"method":"counterfactual","actual":[3,5],"counterfactual":[2]}));
    assert_eq!(counterfactual["result"]["percent_change"], 100.);
    let event = run(json!({"method":"event_study","periods":[0,1,2],
                          "coefficients":[2,3,5],"standard_errors":[1,1,1]}));
    assert_eq!(event["result"]["post_effects"], 4.);
    assert_eq!(event["result"]["event_period_coef"], 2.);
    assert!(event["result"]["pre_trend"].is_null());
    let groups = run(
        json!({"method":"heterogeneous_effects","group_labels":["a","b"],
                           "effects":[2,4],"standard_errors":[1,0]}),
    );
    assert_eq!(groups["result"]["overall_effect"], 3.);
    assert_eq!(groups["result"]["by_group"]["a"]["ci_upper"], 3.96);
    let forecast = run(json!({"method":"counterfactual_forecast","model":"bass",
        "baseline":{"inputs":{"time":[0.,10.]},"parameters":{"p":0.03,"q":0.35,"m":100.}},
        "parameter_overrides":{"m":200.}}));
    let forecast = &forecast["result"];
    assert!(forecast["percentage_difference"][0].is_null());
    assert!((forecast["percentage_difference"][1].as_f64().unwrap() - 100.).abs() < 1e-10);
    let risk = run(json!({"method":"e_value","point_estimate":0.5}));
    assert!((risk["result"]["e_value"].as_f64().unwrap() - (2. + 2_f64.sqrt())).abs() < 1e-12);
    let sensitivity = run(json!({"method":"sensitivity_scenario","point_estimate":10,
                                 "gamma":3,"direction":"negative"}));
    assert_eq!(sensitivity["result"]["lower_bound"], 8.);
    assert_eq!(sensitivity["result"]["upper_bound"], 10.);
    assert_eq!(sensitivity["result"]["method"], "historical_heuristic");
    let grid = run(json!({"method":"sensitivity_grid","point_estimate":-10,"gamma_range":[3,1,2]}));
    assert_eq!(
        grid["result"]["results"],
        json!([
            {"gamma":3.,"robustness":2.},{"gamma":1.,"robustness":0.},{"gamma":2.,"robustness":1.}
        ])
    );
    let paired = run(
        json!({"method":"paired_sensitivity","treated_outcomes":[1,4],
                           "control_outcomes":[0,1],"gamma":2}),
    );
    let upper = paired["result"]["upper_bound"].as_f64().unwrap();
    assert!((upper - (2. + 2_f64.sqrt() * 2_f64.ln())).abs() < 1e-12);
    let balance = run(json!({"method":"covariate_balance","threshold":0.5,
        "treated_covariates":{"shift":[1,3],"same":[1,3],"unmatched":[2,4]},
        "control_covariates":{"shift":[0,2],"same":[1,3]}}));
    let result = &balance["result"];
    let shift = result["smd_by_covariate"]["shift"].as_f64().unwrap();
    assert!((shift - 1. / 2_f64.sqrt()).abs() < 1e-12);
    assert_eq!(result["n_balanced"], 1);
    assert_eq!(result["n_total"], 2);
    assert_eq!(result["all_balanced"], false);
    let invalid = run(
        json!({"method":"heterogeneous_effects","group_labels":["a","a"],
                            "effects":[2,4],"standard_errors":[1,0]}),
    );
    assert_eq!(invalid["error"]["code"], "invalid_request");
}
