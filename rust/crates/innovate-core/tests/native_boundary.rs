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
    let artifact = &rows[0]["artifact"];
    assert_eq!(artifact["model_type"], "logistic");
    assert_eq!(artifact["notes"], "baseline run");
    assert!(artifact["timestamp"].as_str().unwrap().ends_with("+00:00"));
    assert!(artifact["seed"].is_null());
    assert!(rows[1].get("artifact").is_none());
    let mut import = json!({"schema_version":"2.0","operation":"import_scenario_artifact",
        "payload":{"artifact":artifact}});
    assert_eq!(run(&import)["result"]["replay_available"], true);
    let mut seeded = import.clone();
    seeded["payload"]["artifact"]["seed"] = json!(42);
    assert_eq!(run(&seeded)["error"]["code"], "invalid_request");
    let mut modified = import.clone();
    modified["payload"]["artifact"]["replay_request"]["payload"]["workers"] = json!(0);
    assert_eq!(run(&modified)["error"]["code"], "invalid_request");
    import["payload"]["artifact"]["version"] = json!("0.0.1");
    assert_eq!(run(&import)["result"]["replay_available"], false);
    import["payload"]["artifact"]["replay_request"]["payload"]["scenarios"][0]["model_key"] =
        json!("bass");
    assert_eq!(run(&import)["error"]["code"], "invalid_request");
    let replay = run(&artifact["replay_request"]);
    assert_eq!(
        replay["result"]["executions"][0]["result"],
        rows[0]["result"]
    );
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

    let spec = json!({"scenario_type":"baseline","name":"study","description":"",
        "time_horizon":5.,"time_unit":"year","reference_year":2026,
        "market_size":100.,"initial_adoption":0.});
    request["payload"]["scenarios"][0]["specification"] = spec.clone();
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    request["payload"]["scenarios"][1]["specification"] = spec;
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["result"]["time_basis"],
        json!({"time_unit":"year","reference_year":2026})
    );
    request["payload"]["scenarios"][1]["specification"]["time_unit"] = json!("month");
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    request["payload"]["scenarios"][1]["specification"]["time_unit"] = json!("year");
    request["payload"]["scenarios"][1]["specification"]["reference_year"] = json!(2027);
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    request["payload"]["scenarios"][1]["specification"]["reference_year"] = json!(2026);
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
                {"baseline":"c","alternative":"d"}],"interval_mass":0.5,"bootstrap":{"draws":1000,"seed":42}}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let outcome = &response["result"]["comparisons"][0]["result"];
    assert_eq!(outcome["pair_count"], 2);
    assert_eq!(outcome["bootstrap"]["mean_interval"], json!([12.5, 50.]));
    assert_eq!(outcome["bootstrap"]["method"], "paired_percentile");
    assert_eq!(run(&request), response);
    request["payload"]["comparisons"][0]["bootstrap"]["draws"] = json!(10001);
    assert_eq!(
        run(&request)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    request["payload"]["comparisons"][0]["bootstrap"]["draws"] = json!(1000);
    assert_eq!(outcome["median_difference"], 12.5);
    assert_eq!(outcome["mean_difference"], 12.5);
    assert_eq!(outcome["mean_standard_error"], 37.5);
    assert_eq!(outcome["empirical_interval"], json!([-6.25, 31.25]));

    // Enumerating the four equally likely ordered draws from effects
    // {-25, 50} gives means {-25, 12.5, 12.5, 50}. Check quantiles
    // strictly inside the probability masses, away from finite-draw ties.
    request["payload"]["comparisons"][0]["interval_mass"] = json!(0.9);
    request["payload"]["comparisons"][0]["bootstrap"]["draws"] = json!(10_000);
    let expanded = run(&request);
    assert_eq!(
        expanded["result"]["comparisons"][0]["result"]["bootstrap"]["mean_interval"],
        json!([-25., 50.])
    );
    request["payload"]["scenarios"][3]["simulation"]["parameters"]["L"] = json!(400.);
    let constant = run(&request);
    assert_eq!(
        constant["result"]["comparisons"][0]["result"]["bootstrap"]["mean_interval"],
        json!([50., 50.])
    );

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
    assert_eq!(
        rows[0]["specification_translation"]["engines"],
        json!(["kairos_abm", "kairos_des"])
    );
    assert_eq!(
        rows[1]["specification_translation"]["intervention_types"][0],
        "innovation_delta"
    );
    assert_eq!(
        rows[2]["specification_translation"]["model_key"],
        "fixed_pool_replacement"
    );
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

#[test]
fn imports_legacy_scenario_artifacts_without_inventing_replay_inputs() {
    let artifact = json!({"scenario":{"scenario_type":"baseline","name":"baseline",
        "description":"","time_horizon":5,"time_unit":"year","reference_year":2026,
        "market_size":100,"initial_adoption":0},"seed":18446744073709551615_u64,"model_type":"bass",
        "version":"0.6.0","notes":null,"timestamp":"2026-01-01T00:00:00+00:00"});
    let mut request = json!({"schema_version":"2.0","operation":"import_scenario_artifact",
        "payload":{"artifact":artifact}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    assert_eq!(response["result"]["artifact"], artifact);
    assert_eq!(response["result"]["replay_available"], false);
    request["payload"]["artifact"]["timestamp"] = json!("2026-01-01");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn baseline_scenario_supplies_bass_market_and_initial_adopters() {
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
    "payload":{"apply_specifications":true,"include_execution_metadata":true,"scenarios":[{
        "id":"baseline","model_key":"bass",
        "specification":{"scenario_type":"baseline","name":"study","description":"",
            "time_horizon":5,"time_unit":"year","reference_year":2026,
            "market_size":200,"initial_adoption":0.25},
        "simulation":{"inputs":{"time":[0,5]},"parameters":{"p":0,"q":0},
            "simulation_options":{"engine":"kairos_abm","seed":42,"population":20}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([5, 5]));
    assert_eq!(row["result"]["values"], json!([50., 50.]));
    let replay = run(&row["artifact"]["replay_request"]);
    assert_eq!(replay["result"]["executions"][0]["result"], row["result"]);
    request["payload"]["scenarios"][0]["simulation"]["parameters"]["m"] = json!(201);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn bass_intervention_specification_translates_to_kairos_event() {
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
    "payload":{"apply_specifications":true,"include_execution_metadata":true,"scenarios":[{
        "id":"pause","model_key":"bass",
        "specification":{"scenario_type":"intervention","name":"pause","description":"",
            "time_horizon":5,"time_unit":"year","reference_year":2026,"market_size":100,
            "initial_adoption":0,"intervention_type":"innovation_delta",
            "intervention_start_time":0,"intervention_magnitude":-0.2},
        "simulation":{"inputs":{"time":[0,5]},"parameters":{"p":0.2,"q":0},
            "simulation_options":{"engine":"kairos_abm","seed":42,"population":100}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([0, 0]));
    let replay = &row["artifact"]["replay_request"];
    assert_eq!(
        run(replay)["result"]["executions"][0]["result"],
        row["result"]
    );
    request["payload"]["scenarios"][0]["specification"]["intervention_type"] = json!("policy");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn custom_network_specification_checks_topology_and_initial_agents() {
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
    "payload":{"apply_specifications":true,"include_execution_metadata":true,"scenarios":[{
        "id":"network","model_key":"bass",
        "specification":{"scenario_type":"network","name":"custom","description":"",
            "time_horizon":5,"time_unit":"year","reference_year":2026,"market_size":200,
            "initial_adoption":0.25,"network_type":"custom","num_nodes":4,"average_degree":0.5},
        "simulation":{"inputs":{"time":[0,5]},"parameters":{"p":0,"q":0},
            "simulation_options":{"engine":"kairos_network","seed":42,"edges":[[0,1]],
                "initial_agents":[0],"capacity":1,"service_ticks":1}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([1, 1]));
    assert_eq!(row["result"]["values"], json!([50., 50.]));
    assert_eq!(
        run(&row["artifact"]["replay_request"])["result"]["executions"][0]["result"],
        row["result"]
    );
    let mut ring = request.clone();
    let scenario = &mut ring["payload"]["scenarios"][0];
    scenario["specification"]["network_type"] = json!("ring");
    scenario["specification"]["average_degree"] = json!(2.);
    scenario["simulation"]["simulation_options"]
        .as_object_mut()
        .unwrap()
        .remove("edges");
    let response = run(&ring);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([1, 1]));
    let replay = &row["artifact"]["replay_request"];
    assert_eq!(
        replay["payload"]["scenarios"][0]["simulation"]["simulation_options"]["edges"],
        json!([[0, 1], [1, 2], [2, 3], [3, 0]])
    );
    assert_eq!(
        run(replay)["result"]["executions"][0]["result"],
        row["result"]
    );
    let mut random = ring;
    let scenario = &mut random["payload"]["scenarios"][0];
    scenario["specification"]["network_type"] = json!("random_fixed_edges");
    scenario["simulation"]["simulation_options"]["topology_seed"] = json!(77);
    let response = run(&random);
    let row = &response["result"]["executions"][0];
    let metadata = &row["execution_metadata"];
    assert_eq!(
        metadata["submitted_simulation"],
        random["payload"]["scenarios"][0]["simulation"]
    );
    assert_eq!(
        metadata["submitted_simulation"]["simulation_options"]["topology_seed"],
        77
    );
    assert!(
        metadata["simulation"]["simulation_options"]
            .get("topology_seed")
            .is_none()
    );
    assert_eq!(
        row["artifact"]["submitted_simulation"],
        metadata["submitted_simulation"]
    );
    let mut import = json!({"schema_version":"2.0","operation":"import_scenario_artifact",
        "payload":{"artifact":row["artifact"]}});
    assert_eq!(run(&import)["result"]["replay_available"], true);
    import["payload"]["artifact"]["submitted_simulation"]["simulation_options"]
        .as_object_mut()
        .unwrap()
        .remove("topology_seed");
    assert_eq!(run(&import)["error"]["code"], "invalid_request");
    let replay = &row["artifact"]["replay_request"];
    let edges = replay["payload"]["scenarios"][0]["simulation"]["simulation_options"]["edges"]
        .as_array()
        .unwrap();
    assert_eq!(edges.len(), 4);
    assert!(edges.windows(2).all(|pair| pair[0] != pair[1]));
    assert_eq!(
        run(&random)["result"]["executions"][0]["artifact"]["replay_request"],
        *replay
    );
    assert_eq!(
        run(replay)["result"]["executions"][0]["result"],
        row["result"]
    );
    let mut scale_free = random.clone();
    let spec = &mut scale_free["payload"]["scenarios"][0]["specification"];
    spec["network_type"] = json!("scale_free");
    spec["average_degree"] = json!(1.5);
    let output = run(&scale_free);
    let row = &output["result"]["executions"][0];
    let replay = &row["artifact"]["replay_request"];
    let edges = replay["payload"]["scenarios"][0]["simulation"]["simulation_options"]["edges"]
        .as_array()
        .unwrap();
    assert_eq!(edges.len(), 3);
    for node in 1..4 {
        assert_eq!(edges.iter().filter(|edge| edge[1] == node).count(), 1);
    }
    assert_eq!(
        run(replay)["result"]["executions"][0]["result"],
        row["result"]
    );
    random["payload"]["scenarios"][0]["specification"]["average_degree"] = json!(2.1);
    assert_eq!(run(&random)["error"]["code"], "invalid_request");
    request["payload"]["scenarios"][0]["specification"]["average_degree"] = json!(1);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn competition_specification_allocates_initial_adopters_in_competitor_order() {
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
    "payload":{"apply_specifications":true,"include_execution_metadata":true,"scenarios":[{
        "id":"competition","model_key":"multi_product",
        "specification":{"scenario_type":"competition","name":"study","description":"",
            "time_horizon":5,"time_unit":"year","reference_year":2026,"market_size":200,
            "initial_adoption":0.5,"num_competitors":2,"competitor_names":["a","b"],
            "market_share_initial":[0.25,0.75]},
        "simulation":{"inputs":{"time":[0,5]},"parameters":{"p1":0,"p2":0,"q1":0,
            "q2":0,"m1":100,"m2":100,"alpha_1_2":0,"alpha_2_1":0}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["rows"], json!([[25., 75.], [25., 75.]]));
    assert_eq!(
        run(&row["artifact"]["replay_request"])["result"]["executions"][0]["result"],
        row["result"]
    );
    request["payload"]["scenarios"][0]["simulation"]["constructor_kwargs"] =
        json!({"n_products":3});
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn substitution_scenario_conserves_the_adopter_pool_and_replays() {
    let request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
    "payload":{"apply_specifications":true,"include_execution_metadata":true,"scenarios":[{
        "id":"replacement","model_key":"fixed_pool_replacement",
        "specification":{"scenario_type":"substitution","name":"study","description":"",
            "time_horizon":5,"time_unit":"year","reference_year":2026,"market_size":200,
            "initial_adoption":0.5,"incumbent_name":"a","entrant_name":"b","substitution_rate":0.5},
        "simulation":{"inputs":{"time":[0,2]}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["rows"][0], json!([100., 0.]));
    let incumbent = row["result"]["rows"][1][0].as_f64().unwrap();
    let entrant = row["result"]["rows"][1][1].as_f64().unwrap();
    assert!((incumbent - 100. * (-1_f64).exp()).abs() < 1e-12);
    assert!((incumbent + entrant - 100.).abs() < 1e-12);
    assert_eq!(
        run(&row["artifact"]["replay_request"])["result"]["executions"][0]["result"],
        row["result"]
    );
}

#[test]
fn threshold_sensitivity_counts_equal_outcomes_and_preserves_threshold_order() {
    let mut request = json!({"schema_version":"2.0","operation":"threshold_sensitivity",
        "payload":{"outcome_values":[3.,-1.,3.,0.],
            "inputs":[{"name":"adoption","metric":"level","thresholds":[3.,0.,4.]}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let result = &response["result"];
    assert_eq!(
        result["outcome_summary"],
        json!({"min":-1.,"max":3.,"mean":1.25})
    );
    assert_eq!(result["rows"][0]["n_meet"], 2);
    assert_eq!(result["rows"][0]["share_meet"], 0.5);
    assert_eq!(result["rows"][1]["n_meet"], 3);
    assert_eq!(result["rows"][2]["n_meet"], 0);
    request["payload"]["outcome_values"] = json!([]);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn parameter_sensitivity_perturbs_independently_and_reports_elasticity() {
    let request = json!({"schema_version":"2.0","operation":"parameter_sensitivity",
        "payload":{"model_key":"logistic","prediction":{"inputs":{"time":[0.,1.]},
            "parameters":{"L":100.,"k":1.,"x0":1.}},
            "inputs":[{"name":"L","deltas":[20.,-20.]},{"name":"k","deltas":[1.]}]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let rows = response["result"]["rows"].as_array().unwrap();
    assert_eq!(rows[0]["outcome"], 60.);
    assert_eq!(rows[0]["elasticity"], 1.);
    assert_eq!(rows[1]["outcome"], 40.);
    assert_eq!(rows[2]["outcome"], 50.);
}

#[test]
fn assumption_sensitivity_evaluates_explicit_event_options() {
    let mut request = json!({"schema_version":"2.0","operation":"assumption_sensitivity",
        "payload":{"model_key":"logistic","prediction":{"inputs":{"time":[2.]},
            "parameters":{"L":100.,"k":1.,"x0":2.,"L_post":200.,"k_post":1.,"x0_post":2.},
            "constructor_kwargs":{"t_event":2.}},
            "inputs":[{"name":"t_event","alternatives":[3.,1.]}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let rows = response["result"]["rows"].as_array().unwrap();
    assert_eq!(rows[0]["baseline_outcome"], 100.);
    assert_eq!(rows[0]["outcome"], 50.);
    assert_eq!(rows[0]["absolute_change"], -50.);
    assert_eq!(rows[1]["outcome"], 100.);
    request["payload"]["inputs"][0]["name"] = json!("unknown");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn timing_sensitivity_applies_offsets_to_the_same_baseline() {
    let mut request = json!({"schema_version":"2.0","operation":"intervention_timing",
        "payload":{"model_key":"logistic","prediction":{"inputs":{"time":[2.]},
            "parameters":{"L":100.,"k":1.,"x0":2.,"L_post":200.,"k_post":1.,"x0_post":2.},
            "constructor_kwargs":{"t_event":2.}},
            "inputs":[{"name":"t_event","offsets":[1.,-1.,0.]}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let rows = response["result"]["rows"].as_array().unwrap();
    assert_eq!(rows[0]["time"], 3.);
    assert_eq!(rows[0]["absolute_change"], -50.);
    assert_eq!(rows[1]["time"], 1.);
    assert_eq!(rows[2]["absolute_change"], 0.);
    request["payload"]["inputs"][0]["offsets"] = json!([-3.]);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn native_explainability_restores_four_descriptive_summaries() {
    let run = |payload| -> serde_json::Value {
        let request = json!({"schema_version":"2.0","operation":"summarize_explainability","payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let drivers = run(
        json!({"method":"adoption_drivers","drivers":{"price":-1,"access":3},
        "baseline_adoption":10,"scenario_adoption":18}),
    );
    assert_eq!(
        drivers["result"]["contributions"]["access"]["contribution"],
        6.
    );
    let competition = run(
        json!({"method":"competition_effects","product_shares":{"a":0.5,"b":0.25},
        "focal_product":"a"}),
    );
    assert_eq!(competition["result"]["lead"], 0.25);
    let substitution = run(
        json!({"method":"substitution_thresholds","shares":[0.,0.5,0.25],
        "thresholds":[0.5,0.75]}),
    );
    assert_eq!(substitution["result"]["crossings"][0]["first_index"], 1);
    assert!(substitution["result"]["crossings"][1]["first_index"].is_null());
    let policy =
        run(json!({"method":"policy_components","components":{"b":-3,"a":3},"total_effect":2}));
    assert_eq!(policy["result"]["dominant_component"], "a");
    assert_eq!(policy["result"]["residual"], 2.);
    assert_eq!(
        run(json!({"method":"policy_components","components":{}}))["error"]["code"],
        "invalid_request"
    );
}

#[test]
fn combined_reports_preserve_order_and_require_explicit_determinism() {
    let run = |operation, payload| -> serde_json::Value {
        let request = json!({"schema_version":"2.0","operation":operation,"payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let threshold = run(
        "threshold_sensitivity",
        json!({"outcome_values":[1,2],
        "inputs":[{"name":"adoption","metric":"level","thresholds":[2]}]}),
    );
    let policy = run(
        "summarize_explainability",
        json!({"method":"policy_components",
        "components":{"access":2}}),
    );
    let blocks = json!([threshold["result"], policy["result"]]);
    let result = run("combine_report_blocks", json!({"blocks":blocks}));
    assert_eq!(
        result["result"],
        json!({"blocks":blocks,"deterministic":true})
    );
    let mixed = run(
        "combine_report_blocks",
        json!({"blocks":[blocks[0],{"kind":"external"}]}),
    );
    assert_eq!(mixed["result"]["deterministic"], false);
    let invalid = run(
        "combine_report_blocks",
        json!({"blocks":[{"deterministic":"true"}]}),
    );
    assert_eq!(invalid["error"]["code"], "invalid_request");
}

#[test]
fn decision_report_exports_claim_boundaries_and_rejects_unsupported_wording() {
    let run = |payload| -> serde_json::Value {
        let request =
            json!({"schema_version":"2.0","operation":"export_decision_report","payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let mut payload = json!({"title":"Policy scenario","workflow":"policy",
        "claims":[{"claim_type":"Simulation","statement":"Adoption rises in the simulated scenario."}],
        "sensitivity":{"blocks":[{"kind":"threshold_sensitivity"}]}});
    let response = run(payload.clone());
    let report = &response["result"];
    assert_eq!(report["schema_version"], "1.0");
    assert_eq!(report["claim_safety"]["claim_counts"]["simulation"], 1);
    assert_eq!(report["claim_safety"]["recommendations_allowed"], false);
    assert_eq!(report["sensitivity"], payload["sensitivity"]);
    payload["claims"][0]["claim_type"] = json!("causal");
    assert_eq!(run(payload.clone())["error"]["code"], "invalid_request");
    payload["claims"][0]["assumptions"] = json!(["exchangeability"]);
    assert_eq!(
        run(payload.clone())["result"]["claim_safety"]["contains_causal_claims"],
        true
    );
    payload["claims"][0]["statement"] = json!("You should implement this policy.");
    assert_eq!(run(payload)["error"]["code"], "invalid_request");
}

#[test]
fn decision_report_markdown_preserves_boundaries_and_escapes_user_text() {
    let mut request = json!({"schema_version":"2.0","operation":"export_decision_report_markdown",
        "payload":{"title":"Policy scenario","workflow":"policy","claims":[
            {"claim_type":"simulation","statement":"Scenario <example>","evidence_refs":["case-1"]}],
            "diagnostics":{"score":1}}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let markdown = response["result"]["markdown"].as_str().unwrap();
    assert!(markdown.starts_with("# Policy scenario\n\n"));
    assert!(markdown.contains("Scenario \\<example\\>"));
    assert!(markdown.contains("**Evidence references**\n\n- case-1"));
    assert!(markdown.contains("## Diagnostics\n\n```json"));
    assert!(markdown.contains("Recommendations allowed `false`"));
    request["payload"]["claims"][0]["claim_type"] = json!("causal");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn model_cards_distinguish_native_capabilities_from_historical_benchmarks() {
    let run = |payload| -> serde_json::Value {
        let request =
            json!({"schema_version":"2.0","operation":"discover_model_cards","payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let result = run(json!({"key":"bass"}));
    let card = &result["result"]["cards"][0];
    assert_eq!(
        card["native_capability"]["supported_backends"],
        json!(["rust"])
    );
    assert_eq!(
        card["historical_benchmark_card"]["benchmark_case_ids"],
        json!(["bass_smoke_adoption"])
    );
    let result = run(json!({}));
    let cards = result["result"]["cards"].as_array().unwrap();
    assert_eq!(
        cards.len(),
        KernelBinding::new().discover_models_native().models.len()
    );
    assert_eq!(
        cards
            .iter()
            .filter(|card| card["historical_benchmark_card"].is_object())
            .count(),
        10
    );
    assert!(
        cards
            .iter()
            .all(|card| card["historical_benchmark_card"].is_object()
                || card["native_model_card"].is_object())
    );
    let network = run(json!({"key":"network_diffusion"}));
    let card = &network["result"]["cards"][0];
    assert!(card["historical_benchmark_card"].is_null());
    assert_eq!(
        card["native_model_card"]["native_test_target"],
        "network_diffusion"
    );
    assert_eq!(
        card["native_model_card"]["benchmark_case_ids"],
        json!(["network_diffusion_smoke", "network_spillover_reference"])
    );
    assert_eq!(
        run(json!({"key":"unknown"}))["error"]["code"],
        "invalid_request"
    );
}

#[test]
fn native_benchmark_runs_the_historical_synthetic_corpus() {
    let run = |operation, payload| -> serde_json::Value {
        let request = json!({"schema_version":"2.0","operation":operation,"payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let discovery = run("discover_benchmark_cases", json!({}));
    let cases = discovery["result"]["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    let suite = run("run_benchmark_suite", json!({"workers":2}));
    assert_eq!(suite["result"]["succeeded"], 12, "{suite}");
    assert_eq!(suite["result"]["failed"], 0);
    let rows = suite["result"]["runs"].as_array().unwrap();
    for (case, row) in cases.iter().zip(rows) {
        assert_eq!(row["case_id"], case["case_id"]);
        let result = &row["result"];
        assert_eq!(&result["case"], case);
        let sample_count = result["observed"].as_array().unwrap().len();
        assert!(sample_count > 0);
        let key = if matches!(
            case["family"].as_str(),
            Some("competition" | "network_diffusion")
        ) {
            "rows"
        } else {
            "values"
        };
        assert_eq!(
            result["prediction"][key].as_array().unwrap().len(),
            sample_count
        );
        if key == "values" && row["case_id"] != "latent_process_fit" {
            let observed = result["observed"].as_array().unwrap();
            for (prediction, reference) in result["prediction"][key]
                .as_array()
                .unwrap()
                .iter()
                .zip(observed)
            {
                let reference = reference.as_f64().unwrap();
                assert!(
                    (prediction.as_f64().unwrap() - reference).abs()
                        < 1e-5 * reference.abs().max(1.)
                );
            }
        }
        if row["case_id"] == "lock_in_smoke" {
            let observed = result["observed"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap());
            let predicted = result["prediction"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap());
            for (actual, expected) in predicted.zip(observed) {
                assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5);
            }
        }
        if row["case_id"] == "network_diffusion_smoke" {
            assert_eq!(
                result["fit"]["node_diagnostics"].as_array().unwrap().len(),
                2
            );
            let expected = result["observed"][25].as_array().unwrap();
            let actual = result["prediction"]["rows"][25].as_array().unwrap();
            for (a, b) in actual.iter().zip(expected) {
                assert!((a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-5);
            }
        } else if row["case_id"] == "network_spillover_reference" {
            assert!(result["fit"].is_null());
            assert_eq!(
                result["prediction"]["columns"],
                json!(["A", "B", "isolated"])
            );
            for (actual, expected) in result["prediction"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .zip(result["observed"].as_array().unwrap())
            {
                for index in 0..3 {
                    assert!(
                        (actual[index].as_f64().unwrap() - expected[index].as_f64().unwrap()).abs()
                            < 1e-8
                    );
                }
            }
        } else if row["case_id"] == "latent_process_fit" {
            let fit = &result["fit"];
            let params = &fit["parameters"];
            let limit = params["base_L"].as_f64().unwrap();
            let rate = params["base_k"].as_f64().unwrap();
            let center = params["base_x0"].as_f64().unwrap();
            let mut previous = None;
            let mut maximum = 0_f64;
            for (i, time) in result["time"].as_array().unwrap().iter().enumerate() {
                let baseline = limit / (1. + (-rate * (time.as_f64().unwrap() - center)).exp());
                let residual = result["observed"][i].as_f64().unwrap() - baseline;
                let smooth = previous.map_or(residual, |last| 0.4 * residual + 0.6 * last);
                previous = Some(smooth);
                assert!(
                    (fit["state"]["constructor_kwargs"]["latent_values"][i]
                        .as_f64()
                        .unwrap()
                        - smooth)
                        .abs()
                        < 1e-10
                );
                maximum = maximum.max(baseline + smooth);
                assert!(
                    (result["prediction"]["values"][i].as_f64().unwrap() - maximum).abs() < 1e-10
                );
            }
            assert_eq!(
                fit["diagnostics"]["information_criteria"]["available"],
                false
            );
        } else if row["case_id"] == "policy_hazard_fit" {
            assert_eq!(result["fit"]["parameters"]["decay"], 0.3);
            assert_eq!(
                result["fit"]["base_diagnostics"]["optimizer"]["converged"],
                true
            );
        } else if matches!(
            row["case_id"].as_str(),
            Some(
                "policy_hazard_reference"
                    | "latent_process_reference"
                    | "regime_switching_reference"
            )
        ) {
            assert!(result["fit"].is_null());
            for (actual, expected) in result["prediction"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .zip(result["observed"].as_array().unwrap())
            {
                assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-10);
            }
        } else {
            assert!(result["fit"]["diagnostics"].is_object());
        }
    }
}

#[test]
fn benchmark_suite_retains_failures_in_request_order() {
    let request = json!({"schema_version":"2.0","operation":"run_benchmark_suite",
        "payload":{"case_ids":["missing","logistic_growth_smoke"],"workers":2}});
    let result: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(result["result"]["failed"], 1);
    assert_eq!(result["result"]["succeeded"], 1);
    assert_eq!(
        result["result"]["runs"][0]["error"]["code"],
        "benchmark_case_failed"
    );
    assert_eq!(
        result["result"]["runs"][1]["case_id"],
        "logistic_growth_smoke"
    );
}

#[test]
fn benchmark_validation_resolves_all_historical_case_references() {
    let request = json!({"schema_version":"2.0","operation":"validate_benchmark_corpus"});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let report = &response["result"];
    assert_eq!(report["ok"], true);
    assert_eq!(report["summary"]["case_count"], 12);
    assert_eq!(
        report["model_card_summaries"]["bass"]["freshness"]["status"],
        "current"
    );
    assert_eq!(
        report["model_card_summaries"]["bass"]["benchmark_case_versions"]["bass_smoke_adoption"],
        "2026.04"
    );
    assert_eq!(
        report["model_card_summaries"]["lock_in"]["freshness"]["status"],
        "current"
    );
    assert_eq!(report["issues"], json!([]));
    assert_eq!(
        report["model_card_summaries"]["lock_in"]["benchmark_case_versions"]["lock_in_smoke"],
        "native-lock-in-1"
    );
}

#[test]
fn benchmark_ci_policy_selects_fast_cases_without_running_scheduled_fits() {
    let run = |payload| -> serde_json::Value {
        let request =
            json!({"schema_version":"2.0","operation":"run_benchmark_suite","payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(json!({"ci_policy":"fast","workers":2}));
    let rows = response["result"]["runs"].as_array().unwrap();
    let ids: Vec<_> = rows
        .iter()
        .map(|row| row["case_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "bass_smoke_adoption",
            "logistic_growth_smoke",
            "fisher_pry_replacement_smoke"
        ]
    );
    assert_eq!(response["result"]["succeeded"], 3);
    assert_eq!(
        run(json!({"ci_policy":"workflow_dispatch"}))["result"]["runs"][0]["case_id"],
        "network_diffusion_smoke"
    );
    assert_eq!(
        run(json!({"ci_policy":"fast","case_ids":["bass_smoke_adoption"]}))["error"]["code"],
        "invalid_request"
    );
}
