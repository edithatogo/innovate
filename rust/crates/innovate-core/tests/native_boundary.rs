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
    let scenario = run(
        json!({"method":"counterfactual_forecast","forecast_model":"logistic",
        "baseline":{"inputs":{"time":[2.],"covariates":{"price":[0.]}},
            "constructor_kwargs":{"covariates":["price"]},
            "parameters":{"L":100.,"k":0.5,"x0":2.,"beta_L_price":10.,"beta_k_price":0.,"beta_x0_price":0.}},
        "covariate_overrides":{"price":[2.]}}),
    );
    assert_eq!(scenario["result"]["baseline"], json!([50.]));
    assert_eq!(scenario["result"]["counterfactual"], json!([60.]));
    let products = run(
        json!({"method":"counterfactual_forecast","forecast_model":"lock_in",
        "baseline":{"inputs":{"time":[0.]},"constructor_kwargs":{"initial_values":[120.,0.]},
            "parameters":{"alpha1":0.3,"alpha2":0.2,"beta1":0.1,"beta2":0.1,"gamma1":0.1,"gamma2":0.1,"m":100.}},
        "parameter_overrides":{"m":200.}}),
    );
    let products = &products["result"];
    assert_eq!(products["columns"].as_array().unwrap().len(), 2);
    assert_eq!(products["baseline"], json!([[100., 0.]]));
    assert_eq!(products["counterfactual"], json!([[120., 0.]]));
    assert_eq!(products["difference"], json!([[20., 0.]]));
    assert_eq!(products["percentage_difference"], json!([[20., null]]));
    let interval = run(json!({"method":"uncertainty_interval","estimand":"ATE",
        "point_estimate":1.96,"ci_lower":0,"ci_upper":3.92,"n_bootstrap":1000}));
    assert_eq!(interval["result"]["se"], 1.);
    assert_eq!(interval["result"]["excludes_zero"], false);
    assert_eq!(interval["result"]["n_bootstrap"], 1000);
    let reversed = run(json!({"method":"uncertainty_interval","estimand":"ATE",
        "point_estimate":0,"ci_lower":1,"ci_upper":-1}));
    assert_eq!(reversed["error"]["code"], "invalid_request");
    let card = run(
        json!({"method":"model_card","name":"Adoption","description":"Policy comparison",
        "estimand":"ATT","assumptions":["Parallel trends"],"date_created":"2026-11-07"}),
    );
    assert_eq!(card["result"]["version"], "1.0");
    assert_eq!(card["result"]["date_created"], "2026-11-07");
    assert_eq!(card["result"]["assumptions"], json!(["Parallel trends"]));
    let assumption = run(
        json!({"method":"assumption_document","assumption_name":"Parallel trends",
        "mathematical_statement":"Equal expected changes","intuitive_explanation":"Common trend"}),
    );
    assert!(assumption["result"]["assumption"]["how_checked"].is_null());
    let markdown = assumption["result"]["markdown"].as_str().unwrap();
    assert!(markdown.contains("### Parallel trends\n\n"));
    assert!(!markdown.contains("How checked"));
    let evidence = run(json!({"method":"release_evidence","claim":"Effect changed",
        "supporting_evidence":["record"],"caveats":["limitation"]}));
    assert_eq!(evidence["result"]["checks_passed"], false);
    assert_eq!(evidence["result"]["issues"].as_array().unwrap().len(), 1);
    let documented = run(json!({"method":"release_evidence","claim":"Effect changed",
        "supporting_evidence":["record"],"caveats":["limitation"],
        "sensitivity_analysis_conducted":true}));
    assert_eq!(documented["result"]["checks_passed"], true);
    let contrast = run(json!({"method":"treatment_contrast","outcomes":[2,4,7,12],
        "treatment":[0,1,0,1],"effect_modifiers":{"group":["a","a","b","c"]}}));
    assert_eq!(contrast["result"]["overall"]["estimate"], 3.5);
    assert_eq!(
        contrast["result"]["by_modifier"]["group"]["a"]["estimate"],
        2.
    );
    assert!(contrast["result"]["by_modifier"]["group"]["b"]["estimate"].is_null());
    let bootstrap = run(json!({"method":"treatment_contrast","outcomes":[2,2,5,5],
        "treatment":[0,0,1,1],"bootstrap":{"draws":100,"seed":42}}));
    assert_eq!(bootstrap["result"]["bootstrap"]["ci_lower"], 3.);
    assert_eq!(bootstrap["result"]["bootstrap"]["ci_upper"], 3.);
    assert_eq!(
        bootstrap["result"]["bootstrap"]["method"],
        "stratified_percentile"
    );
    let pooled_request = json!({"method":"treatment_contrast","outcomes":[2,5],
        "treatment":[0,1],"bootstrap":{"draws":100,"seed":42,"scheme":"pooled"}});
    let pooled = run(pooled_request.clone());
    assert_eq!(pooled, run(pooled_request));
    let draws = &pooled["result"]["bootstrap"];
    assert_eq!(draws["method"], "pooled_percentile");
    assert_eq!(draws["ci_lower"], 3.);
    assert_eq!(draws["ci_upper"], 3.);
    let discarded = draws["empty_arm_draws"].as_u64().unwrap();
    assert!(discarded > 0);
    assert_eq!(discarded + draws["valid_draws"].as_u64().unwrap(), 100);
    let mut intervention = json!({"method":"intervention_contract","name":"Pilot",
        "timing":"staggered","comparator":"control","start_time":0,"end_time":4,
        "outcome_window":[-1,4],"rollout_schedule":{"region_a":1},"spillover_strength":0.2});
    assert_eq!(
        run(intervention.clone())["result"]["rollout_schedule"]["region_a"],
        1.
    );
    intervention["outcome_window"] = json!([-3, -1]);
    assert_eq!(run(intervention)["error"]["code"], "invalid_request");
    let mut contract = json!({"method":"causal_model_contract","name":"Pilot",
        "treatment_variable":"policy","outcome_variable":"adoption","confounders":["age"],
        "unobserved_confounding_risk":"high"});
    assert_eq!(run(contract.clone())["error"]["code"], "invalid_request");
    contract["has_sensitivity_analysis"] = json!(true);
    assert_eq!(
        run(contract.clone())["result"]["confounders"],
        json!(["age"])
    );
    contract["confounders"] = json!(["Post_Adoption"]);
    assert_eq!(run(contract)["error"]["code"], "invalid_request");
    let mut specification = json!({"method":"policy_specification",
        "intervention":{"name":"Pilot","timing":"post","comparator":"control"},
        "causal_model":{"name":"Evaluation","treatment_variable":"policy",
            "outcome_variable":"adoption","confounders":["age"]},
        "data":{"policy":[0,1],"adoption":[3,5],"age":[20,30]}});
    let exported = run(specification.clone())["result"].clone();
    assert_eq!(exported["n_obs"], 2);
    assert!(exported.get("data").is_none());
    let mut imported = exported.clone();
    imported["method"] = json!("policy_specification");
    assert_eq!(run(imported)["result"], exported);
    let mut unknown = specification.clone();
    unknown["intervention"]["typo"] = json!(true);
    assert_eq!(run(unknown)["error"]["code"], "invalid_request");
    specification["data"]["age"] = json!([20]);
    assert_eq!(run(specification)["error"]["code"], "invalid_request");
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

#[test]
fn dataset_provenance_requires_explicit_license() {
    let run = |license| {
        let request = innovate_core::json!({"schema_version":"2.0","operation":"validate_provenance",
            "payload":{"source":" survey ","license":license,"extraction_time":"2026-11-07T00:00:00Z",
                "transform_steps":[" normalized "]}
        });
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let accepted = run(" CC-BY-4.0 ");
    assert_eq!(accepted["result"]["source"], "survey");
    assert_eq!(accepted["result"]["license"], "CC-BY-4.0");
    assert_eq!(
        accepted["result"]["transform_steps"],
        innovate_core::json!(["normalized"])
    );
    assert_eq!(run(" Unknown ")["error"]["code"], "invalid_request");
}

#[test]
fn dataset_digest_detects_data_changes_but_ignores_extraction_timestamp() {
    let run = |dataset| {
        let request = innovate_core::json!({"schema_version":"2.0","operation":"dataset_checksum",
            "payload":{"dataset":dataset}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    assert_eq!(
        run(innovate_core::json!({"values":[1,2]}))["result"]["checksum"],
        "54fc94706765042e92a6c74fefd93f0134252a9a9ff81fea085dbd55b8227037"
    );
    let mut dataset = innovate_core::json!({"values":[1,2],"provenance":{
        "source":"survey","license":"CC-BY-4.0","extraction_time":"2026-11-07T00:00:00Z"}});
    let digest = run(dataset.clone())["result"]["checksum"].clone();
    assert!(digest.is_string());
    dataset["provenance"]["checksum"] = digest.clone();
    dataset["provenance"]["extraction_time"] = innovate_core::json!("2026-11-08T00:00:00Z");
    assert_eq!(run(dataset.clone())["result"]["checksum"], digest);
    dataset["values"][1] = innovate_core::json!(3);
    assert_eq!(run(dataset)["error"]["code"], "invalid_request");
}

#[test]
fn historical_dataset_digest_matches_python_json_fixture() {
    let request = innovate_core::json!({
        "schema_version":"2.0","operation":"dataset_checksum",
        "payload":{"encoding":"python_sorted_json_v1","dataset":{
            "é":"😀","values":[1,1.0,-0.0,1e-7,1e16,1e-4],"del":"\u{7f}"
        }}
    });
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        response["result"]["checksum"],
        "3655f262bf58ebfcf91ccacfcd757a2584eb6b99b5461efbff2b277c7b223cb1"
    );
    let mut tie = request;
    tie["payload"]["dataset"] = innovate_core::json!({"v":-887777373534812.2});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&tie.to_string())).unwrap();
    assert_eq!(
        response["result"]["checksum"],
        "88e517cfdfe7027d87e6181f2e47690ab024bf1a40de5f94197036f5ab10a543"
    );
}

#[test]
fn adoption_diagnostics_report_multiple_dataset_failures() {
    let request = innovate_core::json!({"schema_version":"2.0","operation":"validate_dataset",
        "payload":{"kind":"adoption","unit":"share","time":[0,0,2],"adoption":[0.2,1.2,0.5],
            "denominator":[1,1,1]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert_eq!(result["ok"], false);
    for name in [
        "time_alignment",
        "duplicate_keys:time",
        "share_bounds:adoption",
        "monotonicity:adoption",
        "denominator_consistency",
    ] {
        assert!(
            result["failed"]
                .as_array()
                .unwrap()
                .contains(&innovate_core::json!(name))
        );
    }
}

#[test]
fn substitution_diagnostics_check_joint_market_share() {
    let request = innovate_core::json!({"schema_version":"2.0","operation":"validate_dataset",
        "payload":{"kind":"substitution","time":[0,1],"share":[[0.5,0.5],[0.7,0.5]]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        response["result"]["failed"],
        innovate_core::json!(["share_sum"])
    );
    let mut labelled = request;
    labelled["payload"]["product_labels"] = innovate_core::json!(["only_one"]);
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&labelled.to_string())).unwrap();
    assert!(
        result["result"]["failed"]
            .as_array()
            .unwrap()
            .contains(&innovate_core::json!("product_labels"))
    );
    labelled["payload"]["schema_version"] = innovate_core::json!("9.0");
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&labelled.to_string())).unwrap();
    assert_eq!(result["error"]["code"], "invalid_request");
}

#[test]
fn panel_and_policy_diagnostics_distinguish_keys_from_event_order() {
    let run = |payload| {
        let request = innovate_core::json!({"schema_version":"2.0","operation":"validate_dataset","payload":payload});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let panel = run(innovate_core::json!({"kind":"competition","time":[0,0,0],
        "value":[1,2,3],"unit_id":["a","a","a"],"product_id":["x","y","x"]}));
    assert_eq!(
        panel["result"]["failed"],
        innovate_core::json!(["duplicate_keys:time_unit_product"])
    );
    let events = run(
        innovate_core::json!({"kind":"policy_timing","event_times":[2,1,1],
        "event_effects":[1,2,3],"event_labels":["x","x","y"]}),
    );
    assert_eq!(events["result"]["ok"], true);
    assert_eq!(
        events["result"]["warnings"],
        innovate_core::json!(["event_time_order"])
    );
}

#[test]
fn edge_diagnostics_detect_duplicate_ordered_pairs_and_negative_weights() {
    let request = innovate_core::json!({"schema_version":"2.0","operation":"validate_dataset",
        "payload":{"kind":"network_edges","source":["a","b","a"],"target":["b","a","b"],
            "weight":[1,0,-1]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        response["result"]["failed"],
        innovate_core::json!(["weight_bounds", "duplicate_keys:source_target"])
    );
}

#[test]
fn dataset_ingestion_stamps_valid_artifacts_and_rejects_invalid_inputs() {
    let run = |dataset| {
        let request = innovate_core::json!({"schema_version":"2.0","operation":"ingest_dataset",
            "payload":{"dataset":dataset}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let mut data = innovate_core::json!({"kind":"adoption","time":[0,1],"adoption":[1,2],
        "provenance":{"source":"survey","license":"CC-BY-4.0","extraction_time":"2026-11-07T00:00:00Z"}});
    let response = run(data.clone());
    let artifact = &response["result"];
    assert_eq!(artifact["artifact_kind"], "innovate.dataset");
    assert_eq!(artifact["validation"]["ok"], true);
    assert_eq!(
        artifact["dataset"]["provenance"]["checksum"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(run(artifact["dataset"].clone())["result"], *artifact);
    let mut changed = artifact["dataset"].clone();
    changed["adoption"][1] = innovate_core::json!(3);
    assert_eq!(run(changed)["error"]["code"], "invalid_request");
    data["adoption"][1] = innovate_core::json!(0);
    let error = run(data)["error"].clone();
    assert_eq!(error["code"], "invalid_request");
    assert_eq!(
        error["validation"]["failed"],
        innovate_core::json!(["monotonicity:adoption"])
    );
}

#[test]
fn scenario_grid_runs_in_order_and_retains_individual_errors() {
    let simulation = json!({
        "inputs":{"time":[1.]},
        "parameters":{"L":100.,"k":1.,"x0":1.}
    });
    let mut request = json!({
        "schema_version":"2.0","operation":"execute_scenario_grid",
        "payload":{"scenarios":[
            {"id":"baseline","model_key":"logistic","simulation":simulation},
            {"id":"broken","model_key":"unknown","simulation":{}},
            {"id":"repeat","model_key":"logistic","simulation":simulation}
        ]}
    });
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let rows = response["result"]["executions"].as_array().unwrap();
    assert_eq!(rows[0]["result"]["values"][0], 50.);
    assert_eq!(rows[1]["error"]["code"], "unsupported_native_operation");
    assert_eq!(rows[2]["id"], "repeat");
    assert_eq!(rows[0]["result"], rows[2]["result"]);

    request["payload"]["workers"] = json!(2);
    assert_eq!(run(&request), response);
    request["payload"]["include_execution_metadata"] = json!(true);
    request["payload"]["scenarios"][0]["notes"] = json!("baseline run");
    let response = run(&request);
    let rows = response["result"]["executions"].as_array().unwrap();
    let metadata = &rows[0]["execution_metadata"];
    assert_eq!(metadata["simulation"], simulation);
    assert_eq!(metadata["notes"], "baseline run");
    assert_eq!(metadata["kernel"]["runtime"], "rust_native");
    assert!(metadata["elapsed_seconds"].as_f64().unwrap() >= 0.);
    assert!(metadata["started_at_unix_seconds"].as_f64().unwrap() > 0.);
    assert!(rows[1]["execution_metadata"]["kernel"].is_null());
    request["payload"]["include_execution_metadata"] = json!("yes");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
    request["payload"]["include_execution_metadata"] = json!(false);
    request["payload"]["workers"] = json!(33);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
    request["payload"]["workers"] = json!(2);
    request["payload"]["scenarios"][1]["id"] = json!("baseline");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn scenario_grid_validates_and_retains_historical_specs() {
    let base = json!({"name":"study","description":"","time_horizon":5.,
        "time_unit":"year","reference_year":2026,"market_size":100.,"initial_adoption":0.});
    for (details, bad_field, bad_value) in [
        (
            json!({"scenario_type":"baseline"}),
            "time_horizon",
            json!(0),
        ),
        (
            json!({"scenario_type":"intervention","intervention_type":"policy",
            "intervention_start_time":1.,"intervention_magnitude":0.2}),
            "intervention_start_time",
            json!(-1),
        ),
        (
            json!({"scenario_type":"substitution","incumbent_name":"A","entrant_name":"B",
            "substitution_rate":0.4}),
            "substitution_rate",
            json!(1.1),
        ),
        (
            json!({"scenario_type":"competition","num_competitors":2,
            "competitor_names":["A","B"],"market_share_initial":[0.4,0.6]}),
            "market_share_initial",
            json!([0.3, 0.6]),
        ),
        (
            json!({"scenario_type":"network","network_type":"custom","num_nodes":3,
            "average_degree":1.}),
            "num_nodes",
            json!(0),
        ),
    ] {
        let mut spec = base.clone();
        spec.as_object_mut()
            .unwrap()
            .extend(details.as_object().unwrap().clone());
        let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
            "payload":{"scenarios":[{"id":"test","model_key":"logistic","specification":spec,
                "simulation":{"inputs":{"time":[1.]},"parameters":{"L":100.,"k":1.,"x0":1.}}}]}});
        let run = |request: &serde_json::Value| -> serde_json::Value {
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
        };
        let response = run(&request);
        assert_eq!(response["result"]["executions"][0]["specification"], spec);
        assert_eq!(
            response["result"]["executions"][0]["result"]["values"][0],
            50.
        );
        request["payload"]["scenarios"][0]["specification"][bad_field] = bad_value;
        assert_eq!(run(&request)["error"]["code"], "invalid_request");
    }
}

#[test]
fn scenario_grid_compares_aligned_adoption_paths() {
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
        "payload":{"scenarios":[
            {"id":"base","model_key":"logistic","simulation":{"inputs":{"time":[1.,2.]},
                "parameters":{"L":100.,"k":1.,"x0":1.}}},
            {"id":"alt","model_key":"logistic","simulation":{"inputs":{"time":[1.,2.]},
                "parameters":{"L":200.,"k":1.,"x0":1.}}}],
            "comparisons":[{"baseline":"base","alternative":"alt","threshold":75.,"relative_margin":0.1}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["comparisons"][0]["result"];
    assert_eq!(row["difference"][0], 50.);
    let baseline = row["ranking"]["baseline_rank"].as_f64().unwrap();
    assert_eq!(row["ranking"]["metric"], "final_adoption");
    assert_eq!(row["ranking"]["alternative_rank"], baseline * 2.);
    assert_eq!(row["incremental_effect"]["absolute_increase"], baseline);
    assert_eq!(
        row["relative_margin_bounds"]["method"],
        "fixed_relative_margin"
    );
    assert_eq!(
        row["relative_margin_bounds"]["baseline"][0],
        baseline - baseline * 0.1
    );
    request["payload"]["comparisons"][0]["relative_margin"] = json!(1.1);
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    request["payload"]["comparisons"][0]["relative_margin"] = json!(0.1);
    assert_eq!(
        row["threshold_timing"],
        json!({"baseline":null,"alternative":1.})
    );

    request["payload"]["scenarios"][1]["simulation"]["inputs"]["time"] = json!([1., 3.]);
    let response = run(&request);
    assert_eq!(
        response["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    assert!(response["result"]["executions"][1]["result"].is_object());
}

#[test]
fn scenario_grid_summarizes_paired_ensemble_differences() {
    let simulations: Vec<_> = [("a", 100.), ("b", 200.), ("c", 300.), ("d", 250.)]
        .into_iter()
        .map(|(id, limit)| {
            json!({"id":id,"model_key":"logistic",
            "simulation":{"inputs":{"time":[1.]},"parameters":{"L":limit,"k":1.,"x0":1.}}})
        })
        .collect();
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
        "payload":{"workers":2,"scenarios":simulations,
            "comparisons":[{"pairs":[{"baseline":"a","alternative":"b"},
                {"baseline":"c","alternative":"d"}],"interval_mass":0.5}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let outcome = &response["result"]["comparisons"][0]["result"];
    assert_eq!(outcome["pair_count"], 2);
    assert_eq!(outcome["median_difference"], 12.5);
    assert_eq!(outcome["empirical_interval"], json!([-6.25, 31.25]));
    request["payload"]["comparisons"][0]["pairs"][1]["baseline"] = json!("a");
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
}

#[test]
fn scenario_discovery_lists_and_resolves_native_families() {
    let mut request = json!({"schema_version":"2.0","operation":"discover_scenarios","payload":{}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let rows = response["result"]["scenarios"].as_array().unwrap();
    let keys: Vec<_> = rows
        .iter()
        .map(|row| row["key"].as_str().unwrap())
        .collect();
    assert_eq!(
        keys,
        [
            "baseline",
            "intervention",
            "substitution",
            "competition",
            "network"
        ]
    );
    request["payload"]["key"] = json!("network");
    assert_eq!(run(&request)["result"]["scenarios"], json!([rows[4]]));
    request["payload"]["key"] = json!("missing");
    assert_eq!(run(&request)["error"]["code"], "unknown_scenario");
}
