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
    for version in ["999", "invalid"] {
        request.schema_version = version.into();
        assert_eq!(
            KernelBinding::new().invoke(&request).unwrap_err().code,
            "invalid_request"
        );
    }
}

#[test]
fn discovery_does_not_advertise_unported_models() {
    let models = KernelBinding::new().discover_models().unwrap().models;
    assert_eq!(models.len(), 16);
    assert!(models.iter().all(|m| m.supported_backends == ["rust"]));
    let request = json!({"schema_version":"2.0","operation":"discover_model_cards",
        "payload":{"key":"invalid_model_key_123"}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(response["error"]["code"], "invalid_request");
    assert_eq!(
        response["error"]["message"],
        "unknown native model card key"
    );
    assert!(response["result"].is_null());
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
fn batch_preserves_uneven_chunks_across_worker_counts() {
    let datasets: Vec<_> = (0..65)
        .map(|index| {
            if index % 7 == 0 {
                serde_json::Value::Null
            } else {
                json!({"inputs":{"time":[0]},"parameters":{"L":2 * index,"k":1,"x0":0}})
            }
        })
        .collect();
    let run = |workers| -> serde_json::Value {
        let request = json!({"schema_version":"2.0","operation":"predict_batch",
            "model_key":"logistic","payload":{"workers":workers,"datasets":datasets}});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let serial = run(1);
    let rows = serial["result"]["datasets"].as_array().unwrap();
    assert_eq!(rows.len(), 65);
    for (index, row) in rows.iter().enumerate() {
        if index % 7 == 0 {
            assert_eq!(row["error"]["code"], "invalid_request");
        } else {
            assert_eq!(row["result"]["values"], json!([index as f64]));
        }
    }
    assert_eq!(run(3), serial);
    assert_eq!(run(32), serial);
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
fn rolling_average_stays_finite_at_maximum_magnitude() {
    for sign in [-1., 1.] {
        for window in 2..=64 {
            let value = sign * f64::MAX;
            let request = json!({"schema_version":"2.0","operation":"preprocess_series",
                "payload":{"method":"rolling_average","window":window,"values":vec![value;window]}});
            let response: serde_json::Value =
                serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
            let mean = response["result"]["values"][window - 1].as_f64().unwrap();
            assert!((mean / value - 1.).abs() <= f64::EPSILON, "window {window}");
        }
    }
}

#[test]
fn rolling_average_retains_subnormal_window_terms() {
    let tiny = f64::from_bits(1);
    let request = json!({"schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"rolling_average","window":2,"values":[tiny,tiny,0.]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(response["result"]["values"], json!([null, tiny, 0.]));
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
fn sarima_near_unit_ar1_matches_exact_stationary_likelihood() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/sarima_near_unit_ar1.json")).unwrap();
    let request = json!({"schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"sarima_fit","order":[1,0,0],"seasonal_order":[0,0,0,0],
            "values":reference["values"]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let fit = &response["result"];
    assert!(
        (fit["coefficients"][0].as_f64().unwrap() - reference["coefficient"].as_f64().unwrap())
            .abs()
            < 2e-6
    );
    assert!(
        (fit["log_likelihood"].as_f64().unwrap() - reference["log_likelihood"].as_f64().unwrap())
            .abs()
            < 1e-8
    );
}

#[test]
fn sarima_mixed_fits_match_independent_likelihood() {
    for (fixture, seasonal_order, coefficient_tolerance) in [
        (
            include_str!("fixtures/sarima_mixed_ar1_ma1.json"),
            [0, 0, 0, 0],
            2e-5,
        ),
        (
            include_str!("fixtures/sarima_mixed_seasonal.json"),
            [1, 0, 0, 4],
            2e-5,
        ),
        (
            include_str!("fixtures/sarima_near_cancellation.json"),
            [0, 0, 0, 0],
            4e-5,
        ),
        (
            include_str!("fixtures/sarima_joint_cancellation.json"),
            [0, 0, 0, 0],
            3e-5,
        ),
    ] {
        let reference: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let request = json!({"schema_version":"2.0","operation":"preprocess_series",
        "payload":{"method":"sarima_fit","order":[1,0,1],"seasonal_order":seasonal_order,
            "values":reference["values"]}});
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        let fit = &response["result"];
        let actual = fit["coefficients"].as_array().unwrap();
        let expected = reference["coefficients"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs()
                    < coefficient_tolerance
            );
        }
        assert!(
            (fit["log_likelihood"].as_f64().unwrap()
                - reference["log_likelihood"].as_f64().unwrap())
            .abs()
                < 2e-7
        );
    }
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
fn month_start_aggregation_preserves_empty_months() {
    let request = json!({"schema_version":"2.0","operation":"aggregate_dates",
        "payload":{"frequency":"MS","dates":["2024-03-15","2024-01-31","2024-03-01"],
            "values":[3,1,2]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(
        response["result"],
        json!({
            "dates":["2024-01-01","2024-02-01","2024-03-01"],"values":[1.,0.,5.]
        })
    );
}

#[test]
fn calendar_quarters_and_years_preserve_labels_and_gaps() {
    for (frequency, dates, values) in [
        (
            "QS",
            json!(["2023-10-01", "2024-01-01", "2024-04-01", "2024-07-01"]),
            json!([2., 0., 0., 3.]),
        ),
        (
            "QE",
            json!(["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30"]),
            json!([2., 0., 0., 3.]),
        ),
        ("YS", json!(["2023-01-01", "2024-01-01"]), json!([2., 3.])),
        ("YE", json!(["2023-12-31", "2024-12-31"]), json!([2., 3.])),
    ] {
        let request = json!({"schema_version":"2.0","operation":"aggregate_dates",
            "payload":{"frequency":frequency,"dates":["2024-07-01","2023-12-31"],"values":[3,2]}});
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert_eq!(
            response["result"],
            json!({"dates":dates,"values":values}),
            "{frequency}"
        );
    }
}

#[test]
fn fiscal_calendar_buckets_cross_calendar_years() {
    let mut request = json!({"schema_version":"2.0","operation":"aggregate_dates",
        "payload":{"frequency":"YE","fiscal_start_month":7,
            "dates":["2024-06-30","2024-07-01","2026-07-01"],"values":[1,2,3]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    assert_eq!(
        run(&request)["result"],
        json!({
            "dates":["2024-06-30","2025-06-30","2026-06-30","2027-06-30"],"values":[1.,2.,0.,3.]
        })
    );
    request["payload"]["fiscal_start_month"] = json!(13);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
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
    let cancellation = run(json!({"method":"counterfactual",
        "actual":[1e16,1.,-1e16],"counterfactual":[0.]}));
    let effect = cancellation["result"]["effect"].as_f64().unwrap();
    assert!((effect - 1. / 3.).abs() < 1e-15);
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
    let mut saved = json!({"method":"counterfactual_forecast","forecast_model":"logistic",
        "baseline":{"inputs":{"time":[2.]},"state":{"model_key":"logistic",
            "parameters":{"L":100.,"k":0.5,"x0":2.},"constructor_kwargs":{},"predict_kwargs":{}}},
        "parameter_overrides":{"L":120.}});
    let state_forecast = run(saved.clone());
    assert_eq!(state_forecast["result"]["baseline"], json!([50.]));
    assert_eq!(state_forecast["result"]["counterfactual"], json!([60.]));
    saved["parameter_overrides"] = json!({"unknown":120.});
    assert_eq!(run(saved)["error"]["code"], "invalid_request");
    let scenario = run(
        json!({"method":"counterfactual_forecast","forecast_model":"logistic",
        "baseline":{"inputs":{"time":[2.],"covariates":{"price":[0.]}},
            "constructor_kwargs":{"covariates":["price"]},
            "parameters":{"L":100.,"k":0.5,"x0":2.,"beta_L_price":10.,"beta_k_price":0.,"beta_x0_price":0.}},
        "covariate_overrides":{"price":[2.]}}),
    );
    assert_eq!(scenario["result"]["time"], json!([2.]));
    assert_eq!(scenario["result"]["baseline"], json!([50.]));
    assert_eq!(scenario["result"]["counterfactual"], json!([60.]));
    let malformed = run(
        json!({"method":"counterfactual_forecast","forecast_model":"bass",
        "parameter_override":{"m":200.}}),
    );
    assert_eq!(
        malformed["error"]["message"],
        "unknown counterfactual forecast field"
    );
    let conflicting = run(
        json!({"method":"counterfactual_forecast","forecast_model":"bass",
        "model":"logistic"}),
    );
    assert_eq!(
        conflicting["error"]["message"],
        "forecast_model and model must agree when both are supplied"
    );
    let mut history = json!({"method":"counterfactual_forecast","forecast_model":"lotka_volterra",
        "baseline":{"inputs":{"time":[3.]},"state":{"model_key":"lotka_volterra",
            "parameters":{"alpha1":0.2,"beta1":0.,"alpha2":0.1,"beta2":0.,
                "beta_alpha1_price":0.1,"beta_beta1_price":0.,"beta_alpha2_price":0.,"beta_beta2_price":0.},
            "constructor_kwargs":{"initial_values":[0.1,0.2],"covariates":["price"],
                "covariate_history":{"time":[0.,2.],"values":{"price":[0.,2.]}}}}},
        "covariate_history_overrides":{"price":[0.,0.]}});
    let compared = run(history.clone());
    assert!(compared["error"].is_null(), "{compared}");
    for (key, exponent) in [("baseline", -1.0_f64), ("counterfactual", -0.6)] {
        let expected = 1. / (1. + 9. * exponent.exp());
        assert!((compared["result"][key][0][0].as_f64().unwrap() - expected).abs() < 1e-8);
    }
    let mut explicit = history.clone();
    let baseline = explicit["baseline"].as_object_mut().unwrap();
    let state = baseline.remove("state").unwrap();
    baseline.insert("parameters".into(), state["parameters"].clone());
    baseline.insert(
        "constructor_kwargs".into(),
        state["constructor_kwargs"].clone(),
    );
    assert_eq!(run(explicit)["result"], compared["result"]);
    history["covariate_history_overrides"]["price"] = json!([0.]);
    assert_eq!(run(history)["error"]["code"], "invalid_request");
    let mut events = json!({"method":"counterfactual_forecast","forecast_model":"lotka_volterra",
        "baseline":{"inputs":{"time":[2.]},"state":{"model_key":"lotka_volterra",
            "parameters":{"alpha1":0.2,"beta1":0.,"alpha2":0.1,"beta2":0.},
            "constructor_kwargs":{"initial_values":[0.1,0.2],
                "parameter_events":[{"time":1.,"parameters":{"alpha1":0.4}}]}}},
        "parameter_event_override":[]});
    let compared = run(events.clone());
    assert!(compared["error"].is_null(), "{compared}");
    for (key, exponent) in [("baseline", -0.6_f64), ("counterfactual", -0.4)] {
        let expected = 1. / (1. + 9. * exponent.exp());
        assert!((compared["result"][key][0][0].as_f64().unwrap() - expected).abs() < 1e-8);
    }
    let mut explicit = events.clone();
    let baseline = explicit["baseline"].as_object_mut().unwrap();
    let state = baseline.remove("state").unwrap();
    baseline.insert("parameters".into(), state["parameters"].clone());
    baseline.insert(
        "constructor_kwargs".into(),
        state["constructor_kwargs"].clone(),
    );
    assert_eq!(run(explicit.clone())["result"], compared["result"]);
    let schedule = explicit["baseline"]["constructor_kwargs"]
        .as_object_mut()
        .unwrap()
        .remove("parameter_events")
        .unwrap();
    explicit["parameter_event_override"] = schedule;
    let reversed = run(explicit);
    assert_eq!(
        reversed["result"]["baseline"],
        compared["result"]["counterfactual"]
    );
    assert_eq!(
        reversed["result"]["counterfactual"],
        compared["result"]["baseline"]
    );
    events["parameter_event_override"] = json!([{"time":1.,"parameters":{"unknown":0.4}}]);
    assert_eq!(run(events)["error"]["code"], "invalid_request");
    let products = run(
        json!({"method":"counterfactual_forecast","forecast_model":"lock_in",
        "baseline":{"inputs":{"time":[0.]},"constructor_kwargs":{"initial_values":[120.,0.]},
            "parameters":{"alpha1":0.3,"alpha2":0.2,"beta1":0.1,"beta2":0.1,"gamma1":0.1,"gamma2":0.1,"m":100.}},
        "parameter_overrides":{"m":200.}}),
    );
    let products = &products["result"];
    assert_eq!(products["columns"].as_array().unwrap().len(), 2);
    assert_eq!(products["time"], json!([0.]));
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
    let mut adjusted = json!({"method":"treatment_contrast","outcomes":[2,5],
        "treatment":[0,1],"estimator":"matching"});
    assert!(run(adjusted.clone())["result"].is_null());
    adjusted.as_object_mut().unwrap().remove("estimator");
    adjusted["confounders"] = json!(["age"]);
    assert!(run(adjusted)["result"].is_null());
    let mut panel = json!({"method":"panel_event_study","periods":[-2,-1,0],"baseline_index":1,
        "treated":[[1,2,7],[3,4,11]],"control":[[0,1,3],[4,5,7]]});
    let result = run(panel.clone());
    let contrasts = &result["result"]["contrasts"];
    assert_eq!(contrasts[0]["estimate"], 0.);
    assert_eq!(contrasts[1]["reference"], true);
    assert_eq!(contrasts[2]["estimate"], 4.);
    assert!((contrasts[2]["standard_error"].as_f64().unwrap() - 1.).abs() < 1e-12);
    panel["periods"] = json!([-2, -2, 0]);
    assert!(run(panel)["result"].is_null());
    let result = run(json!({"method":"staggered_panel_att","periods":[0,1,2],
        "outcomes":[[0,5,7],[10,11,16]],"adoption_indices":[1,2]}));
    let rows = result["result"]["contrasts"].as_array().unwrap();
    assert_eq!(rows[0]["estimate"], 4.);
    assert_eq!(rows[0]["n_control"], 1);
    assert_eq!(rows[0].get("standard_error"), Some(&json!(null)));
    assert_eq!(rows[1]["estimate"], serde_json::Value::Null);
    assert_eq!(rows[2]["n_control"], 0);
    let placebo = run(json!({"method":"staggered_panel_att","periods":[0,1,2],
        "outcomes":[[0,5,7],[10,11,16],[0,2,4]],"adoption_indices":[1,2,null],
        "include_pre_trends":true}));
    let pre = placebo["result"]["pre_trends"].as_array().unwrap();
    assert_eq!(pre.len(), 1);
    assert_eq!(pre[0]["period"], 0.);
    assert_eq!(pre[0]["baseline_period"], 1.);
    assert_eq!(pre[0]["n_control"], 1);
    assert_eq!(pre[0]["estimate"], 1.);
    let never = run(json!({"method":"staggered_panel_att","periods":[0,1,2],
        "outcomes":[[0,5,7],[10,11,16],[0,2,4]],"adoption_indices":[1,2,null],
        "control_group":"never_treated"}));
    assert_eq!(never["result"]["contrasts"][0]["n_control"], 1);
    assert_eq!(never["result"]["contrasts"][0]["estimate"], 3.);
    let mut long = json!({"method":"staggered_long_panel_att",
        "unit_ids":["z","a","z","a","a","z"],"time":[20,10,10,30,20,30],
        "outcomes":[11,0,10,7,5,16],"adoption_times":{"a":20,"z":30}});
    let result = run(long.clone());
    assert_eq!(result["result"]["unit_order"], json!(["a", "z"]));
    assert_eq!(result["result"]["contrasts"][0]["estimate"], 4.);
    long["time"][0] = json!(10);
    assert!(run(long)["result"].is_null());
    let event_time = &result["result"]["event_time"];
    assert_eq!(event_time[0]["supported_cohort_att"], 4.);
    assert_eq!(event_time[0]["unsupported_cohorts"], 1);
    assert_eq!(
        event_time[1]["supported_cohort_att"],
        serde_json::Value::Null
    );
    let replicated = run(json!({"method":"staggered_panel_att","periods":[0,1,2],
        "outcomes":[[0,5,7],[0,7,9],[10,12,16],[10,12,16]],
        "adoption_indices":[1,1,2,null]}));
    let row = &replicated["result"]["contrasts"][0];
    let onset = &replicated["result"]["event_time"][0];
    assert_eq!(onset["supported_treated_units"], 3);
    assert_eq!(onset.get("standard_error"), Some(&json!(null)));
    let shared_request = json!({"method":"staggered_panel_att","periods":[0,1,2],"include_event_time_covariance":true,
        "simultaneous_bands":{"draws":100,"seed":7},
        "outcomes":[[0,5,5],[0,5,5],[0,1,6],[0,1,6],[0,0,0],[0,2,4]],
        "adoption_indices":[1,1,2,2,null,null]});
    let shared = run(shared_request.clone());
    assert_eq!(shared, run(shared_request));
    let bands = &shared["result"]["simultaneous_bands"];
    assert!((bands["critical_value"].as_f64().unwrap() - 2_f64.sqrt()).abs() < 1e-12);
    let se = shared["result"]["event_time"][0]["standard_error"]
        .as_f64()
        .unwrap();
    let expected = (1. / 48_f64.sqrt() + 1. / 8_f64.sqrt()) * 2_f64.sqrt();
    assert!((se - expected).abs() < 1e-12);
    let covariance = &shared["result"]["event_time_covariance"];
    assert!((covariance[0][0].as_f64().unwrap() - se * se).abs() < 1e-12);
    assert!((covariance[0][1].as_f64().unwrap() - 2. * se).abs() < 1e-12);
    assert_eq!(covariance[0][1], covariance[1][0]);
    assert!((onset["supported_cohort_att"].as_f64().unwrap() - 8. / 3.).abs() < 1e-12);
    let balanced = run(json!({"method":"staggered_panel_att","periods":[0,1,2],
        "outcomes":[[0,5,7],[0,7,9],[10,12,16],[10,12,16]],
        "adoption_indices":[1,1,2,null],"balanced_horizon":1}));
    assert_eq!(
        balanced["result"]["balanced_event_time"]["excluded_cohorts"],
        1
    );
    assert_eq!(
        balanced["result"]["event_time"][0]["supported_cohort_att"],
        4.
    );
    assert_eq!(
        balanced["result"]["event_time"][1]["supported_cohort_att"],
        2.
    );
    assert_eq!(row["estimate"], 4.);
    assert!((row["standard_error"].as_f64().unwrap() - 1.).abs() < 1e-12);
    let mut augmented = json!({"method":"augmented_treatment_contrast","outcomes":[0,4],
        "treatment":[0,1],"propensities":[0.5,0.5],"treated_predictions":[2,2],
        "control_predictions":[0,0]});
    let result = run(augmented.clone());
    assert_eq!(result["result"]["estimate"], 4.);
    assert!(
        (result["result"]["influence_standard_error"]
            .as_f64()
            .unwrap()
            - 2.)
            .abs()
            < 1e-12
    );
    augmented["treated_predictions"] = json!([2]);
    assert!(run(augmented)["result"].is_null());
    let mut clustered = json!({"method":"augmented_treatment_contrast","outcomes":[0,4,0,4],
        "treatment":[0,1,0,1],"propensities":[0.5,0.5,0.5,0.5],
        "treated_predictions":[2,2,2,2],"control_predictions":[0,0,0,0],
        "clusters":["a","b","a","b"]});
    let result = run(clustered.clone());
    assert_eq!(result["result"]["estimate"], 4.);
    assert_eq!(result["result"]["cluster_count"], 2);
    assert!(
        (result["result"]["influence_standard_error"]
            .as_f64()
            .unwrap()
            - 2.)
            .abs()
            < 1e-12
    );
    clustered["clusters"] = json!(["a", "a", "a", "a"]);
    assert!(run(clustered)["result"].is_null());
    let mut cross_fitted = json!({"method":"cross_fitted_stratum_ate","outcomes":[0,4,2,6],
        "treatment":[0,1,0,1],"strata":["a","a","a","a"],"folds":[1,2,2,1]});
    let result = run(cross_fitted.clone());
    assert_eq!(
        result["result"]["control_predictions"],
        json!([2., 0., 0., 2.])
    );
    assert_eq!(
        result["result"]["treated_predictions"],
        json!([4., 6., 6., 4.])
    );
    assert_eq!(result["result"]["contrast"]["estimate"], 4.);
    assert_eq!(
        result["result"]["overlap"],
        json!({"min_training_arm_count":1,"min_training_arm_clusters":null,"min_propensity":0.5,"max_propensity":0.5})
    );
    let mut guarded = cross_fitted.clone();
    guarded["minimum_training_arm_count"] = json!(1);
    assert_eq!(run(guarded.clone())["result"], result["result"]);
    guarded["minimum_training_arm_count"] = json!(2);
    assert_eq!(
        run(guarded.clone())["error"]["message"],
        "a held-out stratum falls below minimum_training_arm_count"
    );
    guarded["minimum_training_arm_count"] = json!(0);
    assert_eq!(
        run(guarded)["error"]["message"],
        "minimum_training_arm_count must be a positive integer"
    );
    let mut cluster_guard = cross_fitted.clone();
    cluster_guard["minimum_training_arm_clusters"] = json!(2);
    assert_eq!(
        run(cluster_guard.clone())["error"]["message"],
        "minimum_training_arm_clusters requires cluster labels"
    );
    cluster_guard["clusters"] = json!(["a", "b", "b", "a"]);
    assert_eq!(
        run(cluster_guard.clone())["error"]["message"],
        "a held-out stratum falls below minimum_training_arm_clusters"
    );
    cluster_guard["minimum_training_arm_clusters"] = json!(1);
    assert_eq!(
        run(cluster_guard)["result"]["overlap"]["min_training_arm_clusters"],
        1
    );
    let mut misspelled = cross_fitted.clone();
    misspelled["minimum_training_arms_count"] = json!(2);
    assert_eq!(
        run(misspelled)["error"]["message"],
        "cross-fitted contrast received an unsupported field"
    );
    let matched_with_fitted_controls = run(json!({
        "method":"matched_treatment_contrast",
        "outcomes":cross_fitted["outcomes"],"treatment":cross_fitted["treatment"],
        "scores":[0,0,1,0],"caliper":0,
        "control_predictions":result["result"]["control_predictions"]
    }));
    assert_eq!(matched_with_fitted_controls["result"]["estimate"], 5.);
    assert_eq!(
        matched_with_fitted_controls["result"]["adjusted_estimate"],
        6.
    );
    let mut fitted_matching = json!({"method":"matched_treatment_contrast",
        "outcomes":cross_fitted["outcomes"],"treatment":cross_fitted["treatment"],
        "scores":[0,0,1,0],"caliper":0,"outcome_variances":[1,1,1,1],
        "fit_controls":{"strata":cross_fitted["strata"],"folds":cross_fitted["folds"]}});
    let fitted_result = run(fitted_matching.clone());
    let fit = &fitted_result["result"];
    assert_eq!(fit["adjusted_estimate"], 6.);
    assert_eq!(
        fit["adjusted_outcome_weights"],
        json!([[0, -1.5], [1, 0.5], [2, 0.5], [3, 0.5]])
    );
    assert!((fit["adjusted_standard_error"].as_f64().unwrap() - 3_f64.sqrt()).abs() < 1e-12);
    fitted_matching["fit_controls"]["strata"][1] = json!("unsupported");
    assert!(run(fitted_matching)["result"].is_null());
    cross_fitted["clusters"] = json!(["a", "b", "b", "a"]);
    let result = run(cross_fitted.clone());
    assert_eq!(result["result"]["contrast"]["cluster_count"], 2);
    assert_eq!(result["result"]["overlap"]["min_training_arm_clusters"], 1);
    assert_eq!(
        result["result"]["contrast"]["cluster_degrees_of_freedom"],
        1
    );
    assert!(
        (result["result"]["contrast"]["influence_standard_error"]
            .as_f64()
            .unwrap()
            - 2.)
            .abs()
            < 1e-12
    );
    cross_fitted["clusters"] = json!(["a", "a", "b", "b"]);
    assert!(run(cross_fitted.clone())["result"].is_null());
    cross_fitted.as_object_mut().unwrap().remove("clusters");
    cross_fitted["folds"] = json!([1, 2, 1, 2]);
    assert!(run(cross_fitted)["result"].is_null());
    let mut heterogeneous = json!({"method":"cross_fitted_stratum_att",
        "outcomes":[0,0,2,10,19,19,19,0,0,2,10,19,19,19],
        "treatment":[0,0,1,0,1,1,1,0,0,1,0,1,1,1],
        "strata":["low","low","low","high","high","high","high","low","low","low","high","high","high","high"],
        "folds":[1,1,1,1,1,1,1,2,2,2,2,2,2,2]});
    let att = run(heterogeneous.clone());
    assert_eq!(att["result"]["contrast"]["estimand"], "att");
    assert!((att["result"]["contrast"]["estimate"].as_f64().unwrap() - 7.25).abs() < 1e-12);
    heterogeneous["method"] = json!("cross_fitted_stratum_ate");
    let ate = run(heterogeneous);
    assert!((ate["result"]["contrast"]["estimate"].as_f64().unwrap() - 6.).abs() < 1e-12);
    let result = run(
        json!({"method":"augmented_treatment_contrast","estimand":"att",
        "outcomes":[1,4],"treatment":[0,1],"propensities":[0.5,0.5],
        "control_predictions":[0,1]}),
    );
    assert_eq!(result["result"]["estimate"], 2.);
    assert!(
        (result["result"]["influence_standard_error"]
            .as_f64()
            .unwrap()
            - 2.)
            .abs()
            < 1e-12
    );
    let mut matched = json!({"method":"matched_treatment_contrast",
        "outcomes":[0,10,12,3,99],"treatment":[0,0,1,1,0],
        "scores":[0,2,2,1,0],"caliper":1});
    let result = run(matched.clone());
    assert_eq!(result["result"]["estimate"], 2.5);
    assert_eq!(
        result["result"]["match_diagnostics"],
        json!({"caliper":1.,"max_score_distance":1.,"matched_treated_fraction":1.,"minimum_matched_caliper_slack":0.,"minimum_runner_up_distance_gap":0.})
    );
    assert_eq!(result["result"]["pairs"][0]["control_index"], 1);
    assert_eq!(result["result"]["pairs"][1]["control_index"], 0);
    assert_eq!(
        result["result"]["control_diagnostics"]["effective_sample_size"],
        2.
    );
    let mut cancelling = matched.clone();
    cancelling["outcome_variances"] = json!([0, 0, 0, 0, 0]);
    cancelling["outcome_covariance_factors"] =
        json!([[-f64::MAX, f64::MAX, -f64::MAX, f64::MAX, 0.]]);
    assert_eq!(
        run(cancelling.clone())["result"]["conditional_standard_error"],
        0.
    );
    cancelling["outcome_covariance_factors"] =
        json!([[-f64::MAX, -f64::MAX, f64::MAX, f64::MAX, 0.]]);
    let overflow = run(cancelling);
    assert!(overflow["result"].is_null());
    assert_eq!(overflow["error"]["code"], "invalid_request");
    let mut adjusted = matched.clone();
    adjusted["control_predictions"] = json!([0, 0, 0, 1, 0]);
    let adjusted_result = run(adjusted.clone());
    assert_eq!(adjusted_result["result"]["estimate"], 2.5);
    assert_eq!(adjusted_result["result"]["adjusted_estimate"], 2.);
    adjusted["control_predictions"] = json!([0]);
    assert!(run(adjusted)["result"].is_null());
    let mut reused = matched.clone();
    reused["scores"][3] = json!(2);
    reused["outcome_variances"] = json!([1, 4, 1, 1, 1]);
    reused["balance_covariates"] = json!({"age":[10,20,30,40,50]});
    let mut bad_balance = reused.clone();
    bad_balance["balance_covariates"]["age"] = json!([1]);
    assert!(run(bad_balance)["result"].is_null());
    let mut correlated = reused.clone();
    correlated["outcome_covariance_factors"] = json!([[7, 7, 7, 7, 7], [0, 0, 2, 2, 0]]);
    let correlated_result = run(correlated.clone());
    assert_eq!(
        correlated_result["result"]["uncertainty_model"],
        json!({"model":"diagonal_plus_factors","factor_count":2})
    );
    let se = correlated_result["result"]["conditional_standard_error"]
        .as_f64()
        .unwrap();
    assert!((se - 8.5_f64.sqrt()).abs() < 1e-12);
    correlated["outcome_covariance_factors"] = json!([[1, 2]]);
    assert!(run(correlated)["result"].is_null());
    let result = run(reused);
    assert_eq!(
        result["result"]["uncertainty_model"],
        json!({"model":"independent_outcomes","factor_count":0})
    );
    assert_eq!(
        result["result"]["outcome_weights"],
        json!([[2, 0.5], [3, 0.5], [1, -1.]])
    );
    let age = &result["result"]["covariate_balance"]["age"];
    assert_eq!(age["treated_mean"], 35.);
    assert_eq!(age["control_mean"], 20.);
    assert_eq!(age["mean_difference"], 15.);
    let error = result["result"]["conditional_standard_error"]
        .as_f64()
        .unwrap();
    assert!((error - 4.5_f64.sqrt()).abs() < 1e-12);
    assert_eq!(
        result["result"]["control_diagnostics"]["unique_controls"],
        1
    );
    assert_eq!(
        result["result"]["control_diagnostics"]["effective_sample_size"],
        1.
    );
    let balance = run(json!({"method":"matched_treatment_contrast",
        "outcomes":[0,0,1,1,1],"treatment":[0,0,1,1,1],
        "scores":[0,1,0,0,1],"caliper":0,"balance_covariates":{"age":[0,30,30,30,30],"constant":[1,1,1,1,1]}}));
    let age = &balance["result"]["covariate_balance"]["age"];
    assert_eq!(age["mean_difference"], 20.);
    assert_eq!(age["before_matching"]["mean_difference"], 15.);
    let sd = 150_f64.sqrt();
    assert!((age["standardized_mean_difference"].as_f64().unwrap() - 20. / sd).abs() < 1e-12);
    assert!(
        (age["before_matching"]["standardized_mean_difference"]
            .as_f64()
            .unwrap()
            - 15. / sd)
            .abs()
            < 1e-12
    );
    assert!(
        balance["result"]["covariate_balance"]["constant"]["standardized_mean_difference"]
            .is_null()
    );
    matched["caliper"] = json!(0.5);
    assert!(run(matched.clone())["result"].is_null());
    matched["allow_partial"] = json!(true);
    let result = run(matched);
    assert_eq!(
        result["result"]["match_diagnostics"],
        json!({"caliper":0.5,"max_score_distance":0.,"matched_treated_fraction":0.5,"minimum_matched_caliper_slack":0.5,"minimum_runner_up_distance_gap":2.})
    );
    assert_eq!(result["result"]["estimand"], "matched_subset_att");
    assert_eq!(result["result"]["estimate"], 2.);
    assert_eq!(result["result"]["unmatched_treated_indices"], json!([3]));
    let mut weighted = json!({"method":"propensity_weighted_contrast",
        "outcomes":[0,10,2,12],"treatment":[0,0,1,1],"propensities":[0.2,0.8,0.2,0.8]});
    let mut fit = json!({"method":"estimate_stratum_propensities",
        "treatment":[0,1,0,1,0,1,1],"strata":["low","high","low","low","high","high","high"]});
    let mut logistic = json!({"method":"fit_logistic_propensities",
        "treatment":[0,1],"covariates":[[-2],[2]],"ridge":0.25 / 3_f64.ln()});
    let fitted = run(logistic.clone());
    let mut cross_fit = json!({"method":"cross_fit_logistic_propensities",
        "treatment":[0,1,1,0],"covariates":[[-2],[2],[2],[-2]],
        "folds":[9,4,9,4],"ridge":logistic["ridge"]});
    let cross_fitted = run(cross_fit.clone());
    for (p, expected) in cross_fitted["result"]["propensities"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0.25, 0.75, 0.75, 0.25])
    {
        assert!((p.as_f64().unwrap() - expected).abs() < 1e-7);
    }
    cross_fit["folds"] = json!([4, 9, 9, 4]);
    assert!(run(cross_fit)["result"].is_null());
    let mut prediction = json!({"method":"predict_logistic_propensities",
        "fit":fitted["result"],"covariates":[[0],[2],[-2]]});
    let predicted = run(prediction.clone());
    let mut scoring = json!({"method":"score_propensities","treatment":[0,1,0],
        "propensities":predicted["result"]["propensities"]});
    let scored = run(scoring.clone());
    assert!((scored["result"]["brier_score"].as_f64().unwrap() - 0.125).abs() < 1e-7);
    let expected_loss = -(0.5_f64.ln() + 2. * 0.75_f64.ln()) / 3.;
    assert!((scored["result"]["log_loss"].as_f64().unwrap() - expected_loss).abs() < 1e-7);
    scoring["propensities"][0] = json!(1);
    assert!(run(scoring)["result"].is_null());
    for (p, expected) in predicted["result"]["propensities"]
        .as_array()
        .unwrap()
        .iter()
        .zip([0.5, 0.75, 0.25])
    {
        assert!((p.as_f64().unwrap() - expected).abs() < 1e-7);
    }
    prediction["fit"]["covariate_scales"] = json!([0]);
    assert!(run(prediction)["result"].is_null());
    let fitted = fitted["result"]["propensities"].as_array().unwrap();
    assert!((fitted[0].as_f64().unwrap() - 0.25).abs() < 1e-7);
    assert!((fitted[1].as_f64().unwrap() - 0.75).abs() < 1e-7);
    logistic["treatment"] = json!([1, 1]);
    assert!(run(logistic)["result"].is_null());
    let result = run(fit.clone());
    assert_eq!(
        result["result"]["propensities"],
        json!([1. / 3., 0.75, 1. / 3., 1. / 3., 0.75, 0.75, 0.75])
    );
    let composed = run(json!({"method":"propensity_weighted_contrast",
        "outcomes":[0,19,0,2,10,19,19],"treatment":fit["treatment"],
        "propensities":result["result"]["propensities"]}));
    assert!((composed["result"]["estimate"].as_f64().unwrap() - 6.).abs() < 1e-12);
    fit["strata"][6] = json!("single_arm");
    assert!(run(fit)["result"].is_null());
    let result = run(weighted.clone());
    assert!((result["result"]["estimate"].as_f64().unwrap() + 4.).abs() < 1e-12);
    let error = result["result"]["fixed_propensity_standard_error"]
        .as_f64()
        .unwrap();
    assert!((error - 20.48_f64.sqrt()).abs() < 1e-12);
    let singleton = run(json!({"method":"propensity_weighted_contrast",
        "outcomes":[0,2],"treatment":[0,1],"propensities":[0.5,0.5]}));
    assert_eq!(
        singleton["result"].get("fixed_propensity_standard_error"),
        Some(&json!(null))
    );
    for arm in ["control", "treated"] {
        let diagnostics = &result["result"]["weight_diagnostics"][arm];
        assert_eq!(diagnostics["observations"], 2);
        assert!((diagnostics["maximum_normalized_weight"].as_f64().unwrap() - 0.8).abs() < 1e-12);
        assert!((diagnostics["effective_sample_size"].as_f64().unwrap() - 25. / 17.).abs() < 1e-12);
    }
    weighted["estimand"] = json!("att");
    let att_result = run(weighted.clone());
    assert_eq!(att_result["result"]["estimand"], "att");
    assert!((att_result["result"]["estimate"].as_f64().unwrap() + 41. / 17.).abs() < 1e-12);
    assert_eq!(
        att_result["result"]["weight_diagnostics"]["treated"]["effective_sample_size"],
        2.
    );
    weighted["propensities"][0] = json!(0);
    assert!(run(weighted)["result"].is_null());
    let mut stratified = json!({"method":"stratified_treatment_contrast",
        "outcomes":[0,0,2,10,19,19,19],"treatment":[0,0,1,0,1,1,1],
        "strata":["low","low","low","high","high","high","high"]});
    let discovery = run(json!({"method":"describe_treatment_contrasts"}));
    let methods = discovery["result"]["contrasts"].as_array().unwrap();
    assert_eq!(methods[0]["adjustment"], "none");
    assert_eq!(methods[1]["method"], stratified["method"]);
    assert_eq!(methods[1]["estimands"], json!(["ate", "att"]));
    assert_eq!(methods[3]["partial_estimand"], "matched_subset_att");
    assert_eq!(
        methods[3]["uncertainty_requires"],
        json!(["outcome_variances"])
    );
    assert_eq!(
        methods[3]["uncertainty_assumptions"],
        json!(["fixed_matched_pairs", "independent_outcomes"])
    );
    assert_eq!(
        methods[4]["required_fields_by_estimand"]["ate"],
        json!(["treated_predictions"])
    );
    assert_eq!(methods[4]["required_fields_by_estimand"]["att"], json!([]));
    assert_eq!(methods[5]["method"], "cross_fitted_stratum_ate");
    assert_eq!(methods[6]["method"], "cross_fitted_stratum_att");
    for method in &methods[5..7] {
        assert_eq!(
            method["optional_fields"],
            json!([
                "clusters",
                "minimum_training_arm_count",
                "minimum_training_arm_clusters"
            ])
        );
        assert_eq!(
            method["overlap_diagnostics"],
            json!([
                "min_training_arm_count",
                "min_training_arm_clusters",
                "min_propensity",
                "max_propensity"
            ])
        );
    }
    let result = run(stratified.clone());
    assert!((result["result"]["estimate"].as_f64().unwrap() - 6.0).abs() < 1e-12);
    assert_eq!(result["result"]["by_stratum"]["low"]["n_control"], 2);
    assert_eq!(
        result["result"].get("conditional_standard_error"),
        Some(&json!(null))
    );
    let replicated = run(json!({"method":"stratified_treatment_contrast",
        "outcomes":[0,2,4,6,10,10,12,12],"treatment":[0,0,1,1,0,0,1,1],
        "strata":["low","low","low","low","high","high","high","high"]}));
    let error = replicated["result"]["conditional_standard_error"]
        .as_f64()
        .unwrap();
    assert!((error - 0.5 * 2_f64.sqrt()).abs() < 1e-12);
    let mut att = stratified.clone();
    att["estimand"] = json!("att");
    let result = run(att.clone());
    assert!((result["result"]["estimate"].as_f64().unwrap() - 7.25).abs() < 1e-12);
    assert_eq!(result["result"]["by_stratum"]["low"]["weight"], 0.25);
    assert_eq!(
        result["result"]["by_stratum"]["low"]["treated_fraction"],
        1. / 3.
    );
    att["strata"][0] = json!("control_only");
    assert_eq!(
        run(att.clone())["result"]["excluded_control_observations"],
        1
    );
    assert_eq!(
        run(att.clone())["result"]["estimate"],
        result["result"]["estimate"]
    );
    att["strata"][6] = json!("treated_only");
    assert!(run(att)["result"].is_null());
    stratified["strata"][6] = json!("unsupported");
    assert!(run(stratified)["result"].is_null());
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
    let mut positional = specification.clone();
    positional["causal_model"] = json!(["Evaluation", "policy", "adoption", ["age"], null, {}]);
    assert_eq!(run(positional)["error"]["code"], "invalid_request");
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
    let positional = json!([
        "survey",
        "CC-BY-4.0",
        "2026-11-07T00:00:00Z",
        [],
        "1.0",
        "",
        "",
        {}
    ]);
    for (operation, payload) in [
        ("validate_provenance", positional.clone()),
        (
            "dataset_checksum",
            json!({"dataset":{"provenance":positional}}),
        ),
    ] {
        let request = json!({"schema_version":"2.0","operation":operation,"payload":payload});
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert_eq!(
            response["error"]["message"],
            "dataset provenance must be an object"
        );
    }
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
    dataset["provenance"]["checksum"] = json!("not-a-digest");
    assert_eq!(
        run(dataset.clone())["error"]["message"],
        "dataset provenance checksum must be a 64-digit SHA-256 hex string"
    );
    dataset["provenance"]["checksum"] = json!(digest.as_str().unwrap().to_ascii_uppercase());
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
    let mut reciprocal = request;
    reciprocal["payload"]["source"] = json!(["a", "b"]);
    reciprocal["payload"]["target"] = json!(["b", "a"]);
    reciprocal["payload"]["weight"] = json!([1, 1]);
    for directed in [true, false] {
        reciprocal["payload"]["directed"] = json!(directed);
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&reciprocal.to_string())).unwrap();
        assert_eq!(response["result"]["ok"], directed);
    }
    reciprocal["payload"]["target"] = json!(["a", "a"]);
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&reciprocal.to_string())).unwrap();
    assert_eq!(response["result"]["ok"], true);
    assert_eq!(
        response["result"]["warnings"],
        json!(["self_loops:kairos_network"])
    );
    let mut invalid = reciprocal.clone();
    invalid["payload"]["source"][1] = json!(" ");
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&invalid.to_string())).unwrap();
    assert_eq!(response["result"]["failed"], json!(["identifier_labels"]));
    assert_eq!(
        response["result"]["warnings"],
        json!(["self_loops:kairos_network"])
    );
    let mut missing_weight = reciprocal.clone();
    missing_weight["payload"]["weight"][0] = serde_json::Value::Null;
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&missing_weight.to_string())).unwrap();
    assert_eq!(response["result"]["failed"], json!(["missingness:weight"]));
    let mut zero_weight = reciprocal.clone();
    zero_weight["payload"]["weight"][0] = json!(0);
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&zero_weight.to_string())).unwrap();
    assert_eq!(response["result"]["ok"], true);
    assert_eq!(
        response["result"]["warnings"],
        json!(["self_loops:kairos_network", "zero_weights:kairos_network"])
    );
    let mut dataset = reciprocal["payload"].clone();
    dataset["provenance"] =
        json!({"source":"survey","license":"CC-BY-4.0","extraction_time":"2026-11-07T00:00:00Z"});
    let ingest = |payload| {
        let request =
            json!({"schema_version":"2.0","operation":"ingest_dataset","payload":payload});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let response = ingest(json!({"dataset":dataset}));
    let artifact = &response["result"];
    assert_eq!(
        artifact["validation"]["warnings"],
        json!(["self_loops:kairos_network"])
    );
    assert_eq!(artifact["dataset"]["target"], dataset["target"]);
    assert_eq!(ingest(json!({"artifact":artifact}))["result"], *artifact);
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
        "provenance":{"source":"survey α","license":"CC-BY-4.0","extraction_time":"2026-11-07T00:00:00Z"}});
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
    let ingest_with_encoding = |dataset: &serde_json::Value, encoding: &str| {
        let request = json!({"schema_version":"2.0","operation":"ingest_dataset",
            "payload":{"dataset":dataset,"encoding":encoding}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let legacy = ingest_with_encoding(&data, "python_sorted_json_v1");
    let legacy = &legacy["result"];
    assert_eq!(legacy["checksum_encoding"], "python_sorted_json_v1");
    assert_ne!(
        legacy["dataset"]["provenance"]["checksum"],
        artifact["dataset"]["provenance"]["checksum"]
    );
    assert_eq!(
        ingest_with_encoding(
            &legacy["dataset"],
            legacy["checksum_encoding"].as_str().unwrap()
        )["result"],
        *legacy
    );
    assert_eq!(
        run(legacy["dataset"].clone())["error"]["message"],
        "dataset content checksum does not match provenance"
    );
    let replay = |payload| {
        let request =
            json!({"schema_version":"2.0","operation":"ingest_dataset","payload":payload});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    assert_eq!(replay(json!({"artifact":legacy}))["result"], *legacy);
    let mut stale = legacy.clone();
    stale["validation"] = json!({"ok":false,"failed":["obsolete"]});
    assert_eq!(replay(json!({"artifact":stale}))["result"], *legacy);
    assert_eq!(
        replay(json!({"artifact":legacy,"encoding":"rust_sorted_json_v1"}))["error"]["code"],
        "invalid_request"
    );
    let mut tampered = legacy.clone();
    tampered["dataset"]["adoption"][1] = json!(3);
    assert_eq!(
        replay(json!({"artifact":tampered}))["error"]["message"],
        "dataset content checksum does not match provenance"
    );
    tampered["dataset"]["provenance"]["checksum"] = json!("");
    assert_eq!(
        replay(json!({"artifact":tampered}))["error"]["code"],
        "invalid_request"
    );
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
    let register = json!({"schema_version":"2.0","operation":"register_extension_manifest",
        "payload":{"manifest":{"name":"scenario-serializer","version":"1","entrypoint":"fixture:decode",
            "stability":"stable","extension_points":["serialization_adapter"]}}});
    assert!(run(&register)["error"].is_null());
    let selection = json!({"name":"scenario-serializer","expected_version":"1","extension_point":"serialization_adapter"});
    let invoke = |artifact: &serde_json::Value| {
        innovate_core::invoke_registered_extension(&selection, artifact, |_, artifact| {
            Ok(artifact.clone())
        })
    };
    assert_eq!(invoke(artifact).unwrap(), run(&import)["result"]);
    let mut extension_import = import.clone();
    extension_import["operation"] = json!("import_extension_artifact");
    let response = run(&extension_import);
    assert_eq!(response["operation"], "import_extension_artifact");
    assert_eq!(response["result"], run(&import)["result"]);
    extension_import["payload"] = json!({"dataset":{"kind":"adoption"}});
    assert_eq!(
        run(&extension_import)["error"]["message"],
        "extension import requires a single artifact object"
    );
    let mut older = artifact.clone();
    older["version"] = json!("0.0.1");
    assert_eq!(invoke(&older).unwrap()["replay_available"], false);
    older["timestamp"] = json!("invalid");
    assert_eq!(
        invoke(&older),
        Err("serialization adapter returned invalid scenario artifact")
    );

    let mut seeded = import.clone();
    seeded["payload"]["artifact"]["seed"] = json!(42);
    assert_eq!(run(&seeded)["error"]["code"], "invalid_request");
    let mut modified = import.clone();
    modified["payload"]["artifact"]["replay_request"]["payload"]["workers"] = json!(0);
    assert_eq!(run(&modified)["error"]["code"], "invalid_request");
    let mut extra_request_field = import.clone();
    extra_request_field["payload"]["artifact"]["replay_request"]["model_key"] = json!("bass");
    assert_eq!(
        run(&extra_request_field)["error"]["message"],
        "artifact replay request contains unsupported fields"
    );
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
    let mut ranked = request.clone();
    let mut tie = ranked["payload"]["scenarios"][1].clone();
    tie["id"] = json!("tie");
    ranked["payload"]["scenarios"]
        .as_array_mut()
        .unwrap()
        .push(tie);
    ranked["payload"]["rank_scenarios"] = json!(["tie", "base", "alt"]);
    let ranked_result = run(&ranked);
    let ranks = ranked_result["result"]["ranking"]["rows"]
        .as_array()
        .unwrap();
    assert_eq!(
        ranks.iter().map(|row| &row["id"]).collect::<Vec<_>>(),
        vec![&json!("tie"), &json!("alt"), &json!("base")]
    );
    assert_eq!(
        ranks.iter().map(|row| &row["place"]).collect::<Vec<_>>(),
        vec![&json!(1), &json!(1), &json!(3)]
    );
    ranked["payload"]["rank_metric"] = json!("adoption_area");
    let area = run(&ranked);
    let values = &area["result"]["executions"][1]["result"]["values"];
    let expected = (values[0].as_f64().unwrap() + values[1].as_f64().unwrap()) / 2.;
    assert_eq!(area["result"]["ranking"]["metric"], "adoption_area");
    assert_eq!(
        area["result"]["ranking"]["rows"][0]["adoption_area"],
        expected
    );
    ranked["payload"]["rank_metric"] = json!("time_to_threshold");
    ranked["payload"]["rank_threshold"] = json!(75.);
    let threshold = run(&ranked);
    assert_eq!(threshold["result"]["ranking"]["threshold"], 75.);
    let rows = &threshold["result"]["ranking"]["rows"];
    assert_eq!(rows[0]["time_to_threshold"], 1.);
    assert_eq!(rows[1]["place"], 1);
    assert!(rows[2]["time_to_threshold"].is_null());
    assert!(rows[2]["place"].is_null());
    ranked["payload"]["rank_threshold"] = json!(1_000.);
    let unreached = run(&ranked);
    assert_eq!(unreached["result"]["ranking"]["threshold"], 1_000.);
    let rows = unreached["result"]["ranking"]["rows"].as_array().unwrap();
    assert_eq!(
        rows.iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["tie", "base", "alt"]
    );
    assert!(
        rows.iter()
            .all(|row| row["place"].is_null() && row["time_to_threshold"].is_null())
    );
    ranked["payload"]["rank_scenarios"] = json!(["base", "missing"]);
    let invalid = run(&ranked);
    assert_eq!(
        invalid["result"]["ranking"]["error"]["code"],
        "invalid_ranking"
    );
    assert_eq!(invalid["result"]["executions"].as_array().unwrap().len(), 3);
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
    assert_eq!(row["ranking"]["baseline_place"], 2);
    assert_eq!(row["ranking"]["alternative_place"], 1);
    assert_eq!(row["ranking"]["tie"], false);
    let mut interpolated = request.clone();
    interpolated["payload"]["comparisons"][0]["threshold"] = json!(60.);
    interpolated["payload"]["comparisons"][0]["interpolate_threshold"] = json!(true);
    let interpolated = run(&interpolated);
    let result = &interpolated["result"]["comparisons"][0]["result"];
    assert_eq!(
        result["threshold_timing"],
        json!({"baseline":2.,"alternative":1.})
    );
    let crossing = &result["interpolated_threshold_timing"];
    assert_eq!(crossing["alternative"], 1.);
    assert!((crossing["baseline"].as_f64().unwrap() - (1. + 10. / (baseline - 50.))).abs() < 1e-12);

    let mut missing_threshold = request.clone();
    let comparison = missing_threshold["payload"]["comparisons"][0]
        .as_object_mut()
        .unwrap();
    comparison.remove("threshold");
    comparison.insert("interpolate_threshold".into(), json!(true));
    let rejected = run(&missing_threshold);
    assert_eq!(
        rejected["result"]["executions"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        rejected["result"]["comparisons"][0]["error"]["message"],
        "interpolate_threshold requires a threshold"
    );

    let mut tied = request.clone();
    tied["payload"]["scenarios"][1]["simulation"]["parameters"]["L"] = json!(100.);
    let tied = run(&tied);
    let ranking = &tied["result"]["comparisons"][0]["result"]["ranking"];
    assert_eq!(ranking["baseline_place"], 1);
    assert_eq!(ranking["alternative_place"], 1);
    assert_eq!(ranking["tie"], true);
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
    request["payload"]["rank_scenarios"] = json!(["base", "alt"]);
    assert_eq!(
        run(&request)["result"]["ranking"]["time_basis"],
        json!({"time_unit":"year","reference_year":2026})
    );
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
            "simulation":{"inputs":{"time":[0.,1.]},"parameters":{"L":limit,"k":1.,"x0":1.}}})
        })
        .collect();
    let mut request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
        "payload":{"workers":2,"scenarios":simulations,
            "comparisons":[{"pairs":[{"baseline":"a","alternative":"b"},
                {"baseline":"c","alternative":"d"}],"interval_mass":0.5,"exact_sign_flip":true,"bootstrap":{"draws":1000,"seed":42}}]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let outcome = &response["result"]["comparisons"][0]["result"];
    assert_eq!(outcome["pair_count"], 2);
    assert_eq!(outcome["evaluation_time"], 1.);
    let mut failed_pair = request.clone();
    failed_pair["payload"]["scenarios"][3]["model_key"] = json!("unknown_model");
    let failed_response = run(&failed_pair);
    let failed_result = &failed_response["result"];
    assert_eq!(
        failed_result["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    assert_eq!(
        failed_result["comparisons"][0]["error"]["message"],
        "alternative scenario \"d\": comparison requires a successful scalar simulation"
    );
    assert!(failed_result["comparisons"][0]["result"].is_null());
    assert!(failed_result["executions"][0]["error"].is_null());
    assert!(failed_result["executions"][3]["error"].is_object());
    let mut unknown = request.clone();
    unknown["payload"]["comparisons"][0]["pairs"][1]["alternative"] = json!("missing");
    assert_eq!(
        run(&unknown)["result"]["comparisons"][0]["error"]["message"],
        "alternative scenario \"missing\": unknown scenario ID"
    );
    let mut path_request = request.clone();
    path_request["payload"]["comparisons"][0]["include_path_summary"] = json!(true);
    path_request["payload"]["comparisons"][0]["include_path_covariance"] = json!(true);
    let path_result = run(&path_request);
    let rows = &path_result["result"]["comparisons"][0]["result"]["path_summary"]["rows"];
    assert_eq!(rows[0]["time"], 0.);
    assert_eq!(rows[1]["time"], 1.);
    let covariance = &path_result["result"]["comparisons"][0]["result"]["path_covariance"];
    let first_error = rows[0]["mean_standard_error"].as_f64().unwrap();
    let last_error = rows[1]["mean_standard_error"].as_f64().unwrap();
    assert!((covariance[1][1].as_f64().unwrap() - last_error.powi(2)).abs() < 1e-10);
    assert!((covariance[0][1].as_f64().unwrap() - first_error * last_error).abs() < 1e-10);
    assert_eq!(covariance[0][1], covariance[1][0]);
    for field in [
        "mean_difference",
        "mean_standard_error",
        "empirical_interval",
    ] {
        assert_eq!(rows[1][field], outcome[field]);
    }
    let mut area = request.clone();
    area["payload"]["comparisons"][0]["evaluation_metric"] = json!("adoption_area");
    let result = run(&area);
    let comparison = &result["result"]["comparisons"][0]["result"];
    assert_eq!(comparison["evaluation_horizon"], json!([0., 1.]));
    let paths = &result["result"]["executions"];
    let value = |scenario: usize, period: usize| {
        paths[scenario]["result"]["values"][period]
            .as_f64()
            .unwrap()
    };
    let expected = ((value(1, 0) - value(0, 0)) + (value(1, 1) - value(0, 1))) / 2.;
    assert_eq!(comparison["differences"][0], expected);
    area["payload"]["comparisons"][0]["evaluation_time"] = json!(0.);
    assert_eq!(
        run(&area)["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    let mut sampled = request.clone();
    sampled["payload"]["comparisons"][0]["evaluation_time"] = json!(0.);
    let selected = run(&sampled);
    let selected = &selected["result"]["comparisons"][0]["result"];
    assert_eq!(selected["method"], "paired_empirical_sample_difference");
    assert_eq!(selected["evaluation_time"], 0.);
    assert!(
        (selected["differences"][0].as_f64().unwrap() - 100. / (1. + 1_f64.exp())).abs() < 1e-12
    );
    sampled["payload"]["comparisons"][0]["evaluation_time"] = json!(0.5);
    let rejected = run(&sampled);
    assert_eq!(
        rejected["result"]["executions"].as_array().unwrap().len(),
        4
    );
    assert_eq!(
        rejected["result"]["comparisons"][0]["error"]["code"],
        "invalid_comparison"
    );
    assert_eq!(outcome["sign_flip"]["assignments"], 4);
    assert_eq!(outcome["sign_flip"]["p_value"], 1.);
    let mut aligned = request.clone();
    aligned["payload"]["comparisons"][0]["pairs"][1] = json!({"baseline":"d","alternative":"c"});
    assert_eq!(
        run(&aligned)["result"]["comparisons"][0]["result"]["sign_flip"]["p_value"],
        0.5
    );
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
    assert_eq!(
        rows[4]["resource_comparison"]["option"],
        "compare_resources"
    );
    assert_eq!(
        rows[0]["checkpoint_resume"]["pause_options"],
        json!(["pause_after_adoptions", "pause_at_tick"])
    );
    assert_eq!(
        rows[4]["checkpoint_resume"]["pause_options"],
        json!(["pause_after_completions", "pause_at_tick"])
    );
    assert!(rows[2].get("checkpoint_resume").is_none());
    assert!(rows[0].get("resource_comparison").is_none());
    assert_eq!(
        rows[4]["resource_comparison"]["ensemble_metrics"],
        json!([
            "resource.occupied_slot_ticks",
            "resource.configured_slot_ticks",
            "resource.idle_slot_ticks",
            "resource.over_capacity_slot_ticks",
            "resource.admitted_wait_ticks",
            "resource.abandoned_wait_ticks",
            "resource.pending_wait_ticks",
            "resource.peak_queue_length",
            "resource.peak_in_service",
            "resource.completed_services",
            "resource.balked_requests",
            "resource.abandoned_requests",
            "resource.waiting_at_horizon",
            "resource.in_service_at_horizon"
        ])
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
    let mut network = request.clone();
    let specification = network["payload"]["artifact"]["scenario"]
        .as_object_mut()
        .unwrap();
    specification.extend(
        json!({"scenario_type":"network","network_type":"custom",
        "num_nodes":2,"average_degree":0})
        .as_object()
        .unwrap()
        .clone(),
    );
    let imported = run(&network);
    assert_eq!(imported["result"]["replay_available"], false);
    assert_eq!(
        imported["result"]["artifact"],
        network["payload"]["artifact"]
    );
    network["payload"]["artifact"]["scenario"]["average_degree"] = json!(2);
    let rejected = run(&network);
    assert_eq!(rejected["error"]["code"], "invalid_request");
    assert_eq!(
        rejected["error"]["message"],
        "average_degree cannot exceed num_nodes minus one"
    );

    let mut blank_unit = request.clone();
    blank_unit["payload"]["artifact"]["scenario"]["time_unit"] = json!(" ");
    assert_eq!(
        run(&blank_unit)["error"]["message"],
        "time_unit must be nonblank"
    );
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
                "initial_agents":[0],"capacity":1,"service_ticks":1,
                "schedule":[{"tick":4,"agent_priorities":[0,2,0,0],"agent_service_ticks":[1,2,3,4]}]}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(
        row["result"]["metadata"]["agent_priorities"],
        json!([0, 2, 0, 0])
    );
    assert_eq!(
        row["result"]["metadata"]["agent_service_ticks"],
        json!([1, 2, 3, 4])
    );
    assert_eq!(
        row["artifact"]["replay_request"]["payload"]["scenarios"][0]["simulation"]["simulation_options"]
            ["schedule"],
        request["payload"]["scenarios"][0]["simulation"]["simulation_options"]["schedule"]
    );
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
    assert_eq!(cases.len(), 20);
    let suite = run("run_benchmark_suite", json!({"workers":2}));
    assert_eq!(suite["result"]["succeeded"], 20, "{suite}");
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
            Some("competition" | "network_diffusion" | "hierarchical" | "composite")
        ) {
            "rows"
        } else {
            "values"
        };
        assert_eq!(
            result["prediction"][key].as_array().unwrap().len(),
            sample_count
        );
        if key == "values"
            && !matches!(
                row["case_id"].as_str(),
                Some("latent_process_fit" | "mixture_gaussian_fit")
            )
        {
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
        if row["case_id"] == "composite_coupled_fit" {
            let fit = &result["fit"];
            assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
            let reference: serde_json::Value =
                serde_json::from_str(include_str!("fixtures/composite_coupled.json")).unwrap();
            for (key, expected) in reference["parameters"].as_object().unwrap() {
                assert!(
                    (fit["parameters"][key].as_f64().unwrap() - expected.as_f64().unwrap()).abs()
                        < 1e-5,
                    "{key}: {fit}"
                );
            }
            for (actual, expected) in result["prediction"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .zip(reference["rows"].as_array().unwrap())
            {
                for index in 0..2 {
                    assert!(
                        (actual[index].as_f64().unwrap() - expected[index].as_f64().unwrap()).abs()
                            < 1e-5
                    );
                }
            }
            continue;
        }
        if row["case_id"] == "composite_fit" {
            let fit = &result["fit"];
            assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
            assert_eq!(
                fit["diagnostics"]["initial_condition"]["excluded_conditioning_rows"],
                1
            );
            for (parameter, expected) in [("p_1", 0.03), ("q_1", 0.35), ("m_1", 100.)] {
                assert!((fit["parameters"][parameter].as_f64().unwrap() - expected).abs() < 1e-5);
            }
            for (row, expected) in result["prediction"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .zip(result["observed"].as_array().unwrap())
            {
                assert!((row[0].as_f64().unwrap() - expected[0].as_f64().unwrap()).abs() < 1e-5);
            }
            continue;
        }
        if case["family"] == "composite" {
            assert!(result["fit"].is_null());
            for (actual, expected) in result["prediction"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .zip(result["observed"].as_array().unwrap())
            {
                assert_eq!(actual.as_array().unwrap().len(), 2);
                for index in 0..2 {
                    assert!(
                        (actual[index].as_f64().unwrap() - expected[index].as_f64().unwrap()).abs()
                            < 1e-6
                    );
                }
            }
            if row["case_id"] == "composite_coupled_reference" {
                assert_eq!(result["prediction"]["rates"], json!(vec![[0., 1.]; 4]));
            }
            continue;
        }
        if row["case_id"] == "mixture_gaussian_fit" {
            let fit = &result["fit"];
            assert_eq!(fit["optimizer"]["converged"], true);
            assert_eq!(fit["predictions"]["values"], result["prediction"]["values"]);
            let history = fit["optimizer"]["objective_history"].as_array().unwrap();
            assert!(
                history
                    .windows(2)
                    .all(|pair| pair[1].as_f64().unwrap() + 1e-8 >= pair[0].as_f64().unwrap())
            );
            for (key, expected) in [("model_0_L", 100.), ("model_1_L", 200.), ("weight_0", 0.5)] {
                assert!((fit["parameters"][key].as_f64().unwrap() - expected).abs() < 0.01);
            }
        }
        if row["case_id"] == "hierarchical_group_fit" {
            let prediction = &result["prediction"];
            for ((row, expected), aggregate) in prediction["rows"]
                .as_array()
                .unwrap()
                .iter()
                .zip(result["observed"].as_array().unwrap())
                .zip(prediction["aggregate_values"].as_array().unwrap())
            {
                let expected = expected.as_array().unwrap();
                assert_eq!(row.as_array().unwrap().len(), 2);
                for (actual, expected) in row.as_array().unwrap().iter().zip(expected) {
                    assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-5);
                }
                let total: f64 = expected.iter().map(|v| v.as_f64().unwrap()).sum();
                assert!((aggregate.as_f64().unwrap() - total).abs() < 1e-5);
            }
        }
        if row["case_id"] == "regime_switching_fit" {
            let fit = &result["fit"];
            assert_eq!(
                fit["details"]["detected_breakpoints"],
                json!([5, 10, 15, 20, 30])
            );
            assert_eq!(fit["parameters"]["changepoint_index"], 5);
            assert_eq!(result["prediction"]["values"], fit["predictions"]["values"]);
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
                    | "mixture_reference"
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
    assert_eq!(report["summary"]["case_count"], 20);
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
    assert_eq!(
        response["metadata"]["kernel_version"],
        env!("CARGO_PKG_VERSION")
    );

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
    let selected = run(json!({"case_ids":[ids[2],"unknown_case",ids[0]],"workers":3}));
    let selected = selected["result"]["runs"].as_array().unwrap();
    assert_eq!(selected.len(), 3);
    assert_eq!(selected[0], rows[2]);
    assert_eq!(selected[1]["case_id"], "unknown_case");
    assert!(selected[1]["error"].is_object());
    assert_eq!(selected[2], rows[0]);
    for workers in [1, 32] {
        assert_eq!(
            run(json!({"ci_policy":"fast","workers":workers}))["result"],
            response["result"]
        );
    }
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

#[test]
fn extension_manifest_validation_normalizes_tiers_and_rejects_duplicate_names() {
    let manifest = json!({"name":"example","version":"1","entrypoint":"example:register",
        "stability":"Beta","extension_points":["diagnostics"]});
    let run = |manifests| {
        let request = json!({"schema_version":"2.0","operation":"validate_extension_manifests",
            "payload":{"manifests":manifests}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let result = run(json!([manifest]));
    assert_eq!(result["result"]["manifests"][0]["stability"], "provisional");
    assert_eq!(result["result"]["manifests"][0]["requires"], json!([]));
    assert_eq!(
        run(json!([manifest, manifest]))["error"]["code"],
        "invalid_request"
    );
    for (alias, expected) in [
        (" STABLE ", "stable"),
        ("stable", "stable"),
        ("provisional", "provisional"),
        ("internal", "internal"),
        ("experimental", "provisional"),
        (" BETA ", "provisional"),
        (" InTeRnAl-OnLy ", "internal"),
    ] {
        let request = json!({"schema_version":"2.0","operation":"describe_stability_tier","payload":{"stability":alias}});
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert_eq!(response["result"]["stability"], expected);
        assert!(!response["result"]["guidance"].as_str().unwrap().is_empty());
    }
    let request = json!({"schema_version":"2.0","operation":"describe_stability_tier","payload":{"stability":"unknown"}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(response["error"]["code"], "invalid_request");
    for (field, value) in [
        ("name", json!(" ")),
        ("version", json!("")),
        ("entrypoint", json!("example")),
        ("extension_points", json!([])),
    ] {
        let mut invalid = manifest.clone();
        invalid[field] = value;
        assert_eq!(
            run(json!([invalid]))["error"]["code"],
            "invalid_request",
            "{field}"
        );
    }
    let mut invalid = manifest;
    invalid["extension_points"] = json!(["unknown"]);
    assert_eq!(run(json!([invalid]))["error"]["code"], "invalid_request");
}

#[test]
fn extension_registry_rejects_duplicate_registration_without_replacement() {
    let run = |operation, payload| {
        let request = json!({"schema_version":"2.0","operation":operation,"payload":payload});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let mut manifest = json!({"name":"registry-fixture","version":"1","entrypoint":"example:register",
        "stability":"experimental","extension_points":["diagnostics"]});
    let result = run("register_extension_manifest", json!({"manifest":manifest}));
    assert_eq!(result["result"]["manifest"]["stability"], "provisional");
    manifest["version"] = json!("2");
    let duplicate = run("register_extension_manifest", json!({"manifest":manifest}));
    assert_eq!(duplicate["error"]["code"], "invalid_request");
    let listed = run("list_extension_manifests", json!({}));
    assert!(
        listed["result"]["manifests"]
            .as_array()
            .unwrap()
            .contains(&result["result"]["manifest"])
    );
    let diagnostics = run(
        "list_extension_manifests",
        json!({"extension_point":"diagnostics"}),
    );
    assert!(
        diagnostics["result"]["manifests"]
            .as_array()
            .unwrap()
            .contains(&result["result"]["manifest"])
    );
    let datasets = run(
        "list_extension_manifests",
        json!({"extension_point":"dataset_provider"}),
    );
    assert!(
        !datasets["result"]["manifests"]
            .as_array()
            .unwrap()
            .contains(&result["result"]["manifest"])
    );
    let invalid = run(
        "list_extension_manifests",
        json!({"extension_point":"unknown"}),
    );
    assert_eq!(invalid["error"]["message"], "unknown extension point");
    let resolve = |version, point| {
        run(
            "resolve_extension_manifest",
            json!({"name":"registry-fixture","expected_version":version,"extension_point":point}),
        )
    };
    assert_eq!(
        resolve("1", "diagnostics")["result"]["manifest"],
        result["result"]["manifest"]
    );
    assert_eq!(
        resolve("2", "diagnostics")["error"]["message"],
        "registered extension version does not match expected_version"
    );
    assert_eq!(
        resolve("1", "dataset_provider")["error"]["message"],
        "extension does not declare the requested extension point"
    );
    let selection =
        json!({"name":"registry-fixture","expected_version":"1","extension_point":"diagnostics"});
    let input = json!([{"name":"convergence","status":"fail","message":"iteration limit"}]);
    let invoked =
        innovate_core::invoke_registered_extension(&selection, &input, |resolved, input| {
            assert_eq!(
                resolved["manifest"],
                resolve("1", "diagnostics")["result"]["manifest"]
            );
            assert_eq!(resolved["extension_point"], "diagnostics");
            Ok(input.clone())
        })
        .unwrap();
    assert_eq!(invoked["ok"], false);
    assert_eq!(invoked["failed"], json!(["convergence"]));
    assert_eq!(invoked["checks"], input);
    assert!(
        innovate_core::invoke_registered_extension(&selection, &input, |_, input| Ok(json!([
            input[0], input[0]
        ])))
        .is_err()
    );
    let mut stale = selection;
    stale["expected_version"] = json!("2");
    assert!(
        innovate_core::invoke_registered_extension(&stale, &input, |_, _| {
            panic!("invalid selection invoked the callback")
        })
        .is_err()
    );
    let remove = |version| {
        run(
            "unregister_extension_manifest",
            json!({"name":"registry-fixture","expected_version":version}),
        )
    };
    assert_eq!(
        remove("2")["error"]["message"],
        "registered extension version does not match expected_version"
    );
    let removed = remove("1");
    assert_eq!(removed["result"]["removed"], true);
    assert_eq!(removed["result"]["manifest"], result["result"]["manifest"]);
    assert_eq!(remove("1")["result"]["removed"], false);
    assert_eq!(
        resolve("1", "diagnostics")["error"]["message"],
        "extension is not registered"
    );

    assert_eq!(
        run("register_extension_manifest", json!({"manifest":manifest}))["result"]["manifest"]["version"],
        "2"
    );
    assert_eq!(remove("1")["error"]["code"], "invalid_request");
    assert_eq!(remove("2")["result"]["removed"], true);
}

#[test]
fn extension_dataset_provider_returns_validated_artifact() {
    let request = json!({"schema_version":"2.0","operation":"register_extension_manifest",
        "payload":{"manifest":{"name":"dataset-fixture","version":"1","entrypoint":"fixture:provide",
            "stability":"stable","extension_points":["dataset_provider"]}}});
    let registered: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(registered["error"].is_null());
    let selection = json!({"name":"dataset-fixture","expected_version":"1","extension_point":"dataset_provider"});
    let data = json!({"kind":"adoption","time":[0,1],"adoption":[1,2],
        "provenance":{"source":"fixture","license":"CC-BY-4.0","extraction_time":"2026-11-07T00:00:00Z"}});
    let artifact =
        innovate_core::invoke_registered_extension(&selection, &data, |_, data| Ok(data.clone()))
            .unwrap();
    assert_eq!(artifact["artifact_kind"], "innovate.dataset");
    assert_eq!(artifact["validation"]["ok"], true);
    assert_eq!(
        innovate_core::invoke_registered_extension(&selection, &json!({}), |_, data| Ok(
            data.clone()
        )),
        Err("dataset provider returned invalid dataset")
    );
    let mut request = request;
    request["payload"]["manifest"]["name"] = json!("serialization-fixture");
    request["payload"]["manifest"]["extension_points"] = json!(["serialization_adapter"]);
    let registered: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(registered["error"].is_null());
    let selection = json!({"name":"serialization-fixture","expected_version":"1","extension_point":"serialization_adapter"});
    let roundtrip =
        innovate_core::invoke_registered_extension(&selection, &artifact, |_, artifact| {
            Ok(artifact.clone())
        })
        .unwrap();
    assert_eq!(roundtrip, artifact);
    let mut tampered = artifact;
    tampered["dataset"]["adoption"][0] = json!(0);
    assert_eq!(
        innovate_core::invoke_registered_extension(&selection, &tampered, |_, artifact| Ok(
            artifact.clone()
        )),
        Err("serialization adapter returned invalid dataset artifact")
    );
}

#[test]
fn bootstrap_scalar_fit_replays_and_accounts_for_failed_draws() {
    let time: Vec<f64> = (0..12).map(f64::from).collect();
    let observed: Vec<_> = time
        .iter()
        .map(|t| 100. / (1. + (-0.5 * (t - 5.)).exp()))
        .collect();
    let request = json!({"schema_version":"2.0","operation":"bootstrap_fit",
        "payload":{"fit_model":"logistic","draws":8,"seed":42,
        "fit":{"inputs":{"time":time,"observed":observed}}}});
    let response = innovate_core::execute_json(&request.to_string());
    assert_eq!(response, innovate_core::execute_json(&request.to_string()));
    let result: serde_json::Value = serde_json::from_str(&response).unwrap();
    let fit = &result["result"];
    let succeeded = fit["succeeded"].as_u64().unwrap();
    assert!(succeeded > 0);
    assert_eq!(succeeded + fit["failed"].as_u64().unwrap(), 8);
    assert_eq!(fit["baseline_optimizer"]["converged"], true);
    let failure_counts: u64 = fit["failure_counts"]
        .as_object()
        .unwrap()
        .values()
        .map(|count| count.as_u64().unwrap())
        .sum();
    assert_eq!(failure_counts, fit["failed"].as_u64().unwrap());
    assert_eq!(fit["interval_method"], "percentile");
    assert_eq!(fit["interval_mass"], 0.95);
    for bound in fit["parameter_intervals"]["L"].as_array().unwrap() {
        assert!((bound.as_f64().unwrap() - 100.).abs() < 1e-3);
    }
    for estimate in fit["estimates"]["L"].as_array().unwrap() {
        assert!((estimate.as_f64().unwrap() - 100.).abs() < 1e-3);
    }
    let estimates: Vec<f64> = serde_json::from_value(fit["estimates"]["L"].clone()).unwrap();
    let mean = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let variance = estimates
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (estimates.len() - 1) as f64;
    assert!(
        (fit["parameter_standard_errors"]["L"].as_f64().unwrap() - variance.sqrt()).abs() < 1e-12
    );
    let mut residual = request;
    residual["payload"]["method"] = json!("centered_residual_resampling");
    let response = innovate_core::execute_json(&residual.to_string());
    assert_eq!(response, innovate_core::execute_json(&residual.to_string()));
    let result: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(result["result"]["method"], "centered_residual_resampling");
    assert_eq!(result["result"]["succeeded"], 8);
    for bound in result["result"]["parameter_intervals"]["L"]
        .as_array()
        .unwrap()
    {
        assert!((bound.as_f64().unwrap() - 100.).abs() < 1e-3);
    }
}

#[test]
fn in_process_remote_dispatch_applies_scope_and_operation_policy() {
    let mut request = json!({"schema_version":"2.0","operation":"execute_remote_in_process",
        "payload":{"schema_version":"2.0","context":{"request_id":"request","tenant_id":"tenant",
            "principal":"synthetic","auth_scope":"kernel:execute","trace_id":"trace"},
            "kernel_request":{"schema_version":"2.0","operation":"predict_model","model_key":"logistic",
            "payload":{"inputs":{"time":[1.]},"parameters":{"L":100.,"k":1.,"x0":1.}}}}});
    let run = |request: &serde_json::Value| {
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let contract =
        run(&json!({"schema_version":"2.0","operation":"describe_remote_execution_contract"}));
    let contract = &contract["result"];
    assert_eq!(
        contract["envelope_fields"],
        json!(["schema_version", "context", "kernel_request"])
    );
    assert_eq!(
        contract["kernel_request_fields"],
        json!([
            "schema_version",
            "operation",
            "model_key",
            "payload",
            "metadata"
        ])
    );
    assert_eq!(contract["max_kernel_request_bytes"], 1_000_000);
    assert_eq!(contract["identity_verification"], "caller_supplied_context");
    assert_eq!(contract["local_only_by_default"], json!(["fit_model"]));
    assert!(
        contract["eligible_operations"]
            .as_array()
            .unwrap()
            .contains(&json!("predict_model"))
    );
    let response = run(&request);
    assert_eq!(response["result"]["status"], "ok");
    assert_eq!(
        response["result"]["kernel_response"]["result"]["values"],
        json!([50.])
    );
    assert_eq!(response["result"]["observability"]["trace_id"], "trace");
    let mut malformed = request.clone();
    malformed["payload"]["context"]["trace_id"] = json!("é".repeat(129));
    assert_eq!(run(&malformed)["error"]["code"], "invalid_request");
    malformed["payload"]["context"]["trace_id"] = json!("line\nbreak");
    assert_eq!(run(&malformed)["error"]["code"], "invalid_request");
    malformed["payload"]["context"]["trace_id"] = json!("é".repeat(128));
    assert_eq!(run(&malformed)["result"]["status"], "ok");
    malformed["payload"]["transport"] = json!("https");
    assert_eq!(
        run(&malformed)["error"]["message"],
        "remote envelope contains unsupported fields"
    );
    let mut unknown_kernel_field = request.clone();
    unknown_kernel_field["payload"]["kernel_request"]["timeout_ms"] = json!(500);
    assert_eq!(
        run(&unknown_kernel_field)["error"]["message"],
        "remote kernel request contains unsupported fields"
    );
    request["payload"]["context"]["auth_scope"] = json!("read");
    assert_eq!(run(&request)["result"]["status"], "error");
    request["payload"]["context"]["auth_scope"] = json!("kernel:execute");
    request["payload"]["kernel_request"]["operation"] = json!("fit_model");
    assert_eq!(
        run(&request)["result"]["kernel_response"]["error"]["code"],
        "unsupported_operation"
    );
    request["payload"]["kernel_request"]["operation"] = json!("fetch_oecd_dataset");
    assert_eq!(run(&request)["error"]["message"], "invalid kernel request");
}

#[test]
fn bayesian_logistic_capacity_grid_matches_symmetric_gaussian_reference() {
    let mut request = json!({"schema_version":"2.0","operation":"bayesian_parameter_grid",
        "payload":{"forecast_model":"logistic","parameter":"L","bounds":[80.,120.],
            "points":401,"noise_sd":1.,"observed":[50.],
            "prediction":{"inputs":{"time":[2.]},"parameters":{"L":100.,"k":0.5,"x0":2.}}}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(response["error"].is_null(), "{response}");
    let fit = &response["result"];
    let mass: Vec<f64> = serde_json::from_value(fit["posterior_mass"].clone()).unwrap();
    let grid: Vec<f64> = serde_json::from_value(fit["grid"].clone()).unwrap();
    assert!((mass.iter().sum::<f64>() - 1.).abs() < 1e-12);
    assert!((fit["posterior_mean"].as_f64().unwrap() - 100.).abs() < 1e-10);
    let interval = fit["credible_interval"].as_array().unwrap();
    assert_eq!(fit["credible_mass"], 0.95);
    assert!((interval[0].as_f64().unwrap() - 96.080072).abs() < 0.1);
    assert!((interval[1].as_f64().unwrap() - 103.919928).abs() < 0.1);
    let variance: f64 = grid
        .iter()
        .zip(mass)
        .map(|(x, p)| p * (x - 100.).powi(2))
        .sum();
    assert!((variance - 4.).abs() < 1e-10);
    assert_eq!(response["schema_version"], "2.0");
    request["payload"]
        .as_object_mut()
        .unwrap()
        .remove("forecast_model");
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(response["error"]["message"], "forecast_model is required");
}

#[test]
fn bayesian_joint_grid_preserves_row_major_mass() {
    let request = json!({"schema_version":"2.0","operation":"bayesian_parameter_grid",
        "payload":{"forecast_model":"logistic","axes":[
            {"parameter":"L","bounds":[80.,120.],"points":5},
            {"parameter":"k","bounds":[0.1,0.9],"points":7}],
            "noise_sd":2.,"observed":[50.,62.],
            "prediction":{"inputs":{"time":[2.,3.]},"parameters":{"L":100.,"k":0.5,"x0":2.}}}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(response["error"].is_null(), "{response}");
    let result = &response["result"];
    assert_eq!(result["shape"], json!([5, 7]));
    let mass = result["posterior_mass"].as_array().unwrap();
    let mut weights = Vec::new();
    for i in 0..5 {
        let capacity = 84. + 8. * f64::from(i);
        for j in 0..7 {
            let k = 0.1 + 0.8 * (f64::from(j) + 0.5) / 7.;
            let residuals = [capacity / 2. - 50., capacity / (1. + (-k).exp()) - 62.];
            weights.push((-residuals.iter().map(|x| x * x).sum::<f64>() / 8.).exp());
        }
    }
    let total = weights.iter().sum::<f64>();
    assert_eq!(mass.len(), weights.len());
    for (actual, expected) in mass.iter().zip(weights) {
        assert!((actual.as_f64().unwrap() - expected / total).abs() < 1e-12);
    }
}

#[test]
fn bayesian_metropolis_replays_and_recovers_logistic_capacity() {
    let request = json!({"schema_version":"2.0","operation":"bayesian_metropolis",
        "payload":{"forecast_model":"logistic","parameter":"L","bounds":[80.,120.],
            "draws":4000,"warmup":1000,"seed":42,"proposal_step":4.,"noise_sd":1.,"observed":[50.],
            "prediction":{"inputs":{"time":[2.]},"parameters":{"L":100.,"k":0.5,"x0":2.}}}});
    let raw = innovate_core::execute_json(&request.to_string());
    assert_eq!(raw, innovate_core::execute_json(&request.to_string()));
    let response: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert!(response["error"].is_null(), "{response}");
    let fit = &response["result"];
    assert_eq!(fit["samples"].as_array().unwrap().len(), 4000);
    assert!((fit["posterior_mean"].as_f64().unwrap() - 100.).abs() < 0.25);
    assert!((0.2..0.9).contains(&fit["acceptance_rate"].as_f64().unwrap()));
}

#[test]
fn bayesian_metropolis_retains_ordered_joint_draws() {
    let execute = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let mut request = json!({"schema_version":"2.0","operation":"bayesian_metropolis",
        "payload":{"forecast_model":"logistic","axes":[
            {"parameter":"L","bounds":[80.,120.],"proposal_step":4.},
            {"parameter":"k","bounds":[0.1,0.9],"proposal_step":0.08}],
            "draws":200,"warmup":50,"seed":42,"noise_sd":1.,"observed":[50.],
            "prediction":{"inputs":{"time":[2.]},"parameters":{"L":100.,"k":0.5,"x0":2.}}}});
    let response = execute(&request);
    let result = &response["result"];
    assert_eq!(result["parameters"], json!(["L", "k"]));
    let rows = result["samples"].as_array().unwrap();
    assert_eq!(rows.len(), 200);
    for row in rows {
        assert!((80.0..=120.0).contains(&row[0].as_f64().unwrap()));
        assert!((0.1..=0.9).contains(&row[1].as_f64().unwrap()));
    }
    assert_ne!(rows.first(), rows.last());
    request["payload"]["axes"][1]["parameter"] = json!("L");
    assert!(execute(&request)["error"].is_object());
}

#[test]
fn bayesian_classical_split_rhat_detects_drift_and_stuck_chains() {
    let diagnose = |chains| {
        let request = json!({"schema_version":"2.0","operation":"bayesian_split_rhat",
            "payload":{"chains":chains}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let drifting = diagnose(json!([[0, 2, 999, 4, 6], [4, 6, 999, 0, 2]]));
    let rhat = drifting["result"]["rhat"].as_f64().unwrap();
    assert!((rhat - (19_f64 / 6.).sqrt()).abs() < 1e-12);
    let stuck = diagnose(json!([[0, 0, 0, 0], [1, 1, 1, 1]]));
    assert_eq!(stuck["result"]["status"], "zero_within_split_variance");
    assert!(stuck["result"]["rhat"].is_null());
    assert!(diagnose(json!([[0, 0, 0, 0], [0, 0, 0]]))["error"].is_object());
}

#[test]
fn bayesian_ranked_split_rhat_detects_chain_scale_difference() {
    let request = json!({"schema_version":"2.0","operation":"bayesian_split_rhat",
        "payload":{"method":"rank_normalized_split_rhat",
            "chains":[[-2,0,0,2,999,-2,0,0,2],[-20,-10,10,20,999,-20,-10,10,20]]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert!((result["bulk_rhat"].as_f64().unwrap() - 0.75_f64.sqrt()).abs() < 1e-12);
    // Independent R oracle using rank(), qnorm(), and sample variances.
    assert!((result["folded_rhat"].as_f64().unwrap() - 1.97502379702702).abs() < 1e-12);
    assert_eq!(result["rhat"], result["folded_rhat"]);
}

#[test]
fn bayesian_effective_sample_size_distinguishes_persistence_and_stuck_chains() {
    let diagnose = |chains| {
        let request = json!({"schema_version":"2.0","operation":"bayesian_effective_sample_size",
            "payload":{"chains":chains}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let persistent: Vec<_> = (0..80).map(|draw| (draw / 10) % 2).collect();
    let alternating: Vec<_> = (0..80).map(|draw| draw % 2).collect();
    let slow = diagnose(json!([persistent, persistent]));
    let fast = diagnose(json!([alternating, alternating]));
    assert_eq!(slow["result"]["draws_used"], 160);
    assert!(slow["result"]["bulk_ess"].as_f64().unwrap() < 80.);
    assert!(fast["result"]["bulk_ess"].as_f64().unwrap() > 160.);
    let stuck = diagnose(json!([vec![0; 80], vec![1; 80]]));
    assert!(stuck["error"].is_null());
    assert!(stuck["result"]["bulk_ess"].is_null());
}

#[test]
fn bayesian_metropolis_chains_diagnoses_each_joint_parameter() {
    let mut request = json!({"schema_version":"2.0","operation":"bayesian_metropolis_chains",
        "payload":{"forecast_model":"logistic","axes":[
            {"parameter":"L","bounds":[80.,120.],"proposal_step":4.},
            {"parameter":"k","bounds":[0.1,0.9],"proposal_step":0.08}],
            "draws":200,"warmup":50,"noise_sd":1.,"observed":[50.],
            "chains":[{"seed":17,"initial":{"L":95.,"k":0.4}},
                      {"seed":42,"initial":{"L":105.,"k":0.6}}],
            "prediction":{"inputs":{"time":[2.]},"parameters":{"x0":2.}}}});
    let execute = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = execute(&request);
    assert_eq!(response, execute(&request));
    let result = &response["result"];
    assert_eq!(result["chains"][0]["initial"]["L"], 95.);
    assert_eq!(result["chains"][1]["initial"]["L"], 105.);
    for (axis, parameter) in ["L", "k"].into_iter().enumerate() {
        let diagnostic = &result["diagnostics"][axis];
        assert_eq!(diagnostic["parameter"], parameter);
        assert!(diagnostic["split_rhat"]["rhat"].is_number());
        assert_eq!(diagnostic["effective_sample_size"]["draws_used"], 400);
    }
    request["payload"]["chains"][1]["seed"] = json!(17);
    assert!(execute(&request)["error"].is_object());
}

#[test]
fn bayesian_gaussian_prior_combines_with_gaussian_likelihood() {
    let mut request = json!({"schema_version":"2.0","operation":"bayesian_parameter_grid",
        "payload":{"forecast_model":"logistic","parameter":"L","bounds":[80.,120.],
            "points":401,"noise_sd":1.,"observed":[50.],
            "gaussian_priors":{"L":{"mean":96.,"sd":2.}},
            "prediction":{"inputs":{"time":[2.]},"parameters":{"L":100.,"k":0.5,"x0":2.}}}});
    let execute = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = execute(&request);
    let result = &response["result"];
    assert!((result["posterior_mean"].as_f64().unwrap() - 98.).abs() < 1e-10);
    request["payload"]["gaussian_priors"]["L"]["sd"] = json!(0);
    assert!(execute(&request)["error"].is_object());
    request["payload"]["gaussian_priors"] = json!({"k":{"mean":0.5,"sd":1}});
    assert!(execute(&request)["error"].is_object());
}

#[test]
fn kairos_scenario_replay_retains_scheduled_retry_clear() {
    let request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
    "payload":{"apply_specifications":true,"include_execution_metadata":true,"scenarios":[{
        "id":"retry","model_key":"bass",
        "specification":{"scenario_type":"network","name":"retry","description":"",
            "time_horizon":12,"time_unit":"year","reference_year":2026,"market_size":2,
            "initial_adoption":0,"network_type":"custom","num_nodes":2,"average_degree":1},
        "simulation":{"inputs":{"time":[0,6,12]},"parameters":{"p":1e12,"q":0},
            "simulation_options":{"engine":"kairos_network","seed":42,"edges":[[0,1]],
                "capacity":1,"service_ticks":5,"ticks_per_unit":1,"max_queue_length":0,
                "retry":{"delay_ticks":4,"max_retries":1},"schedule":[{"tick":3,"retry":null}]}}
    }]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([0, 1, 2]), "{response}");
    assert_eq!(row["result"]["retry_eligibility"][0]["tick"], 6);
    assert_eq!(
        row["result"]["resource_metrics"],
        json!({
            "horizon_ticks":12,"occupied_slot_ticks":10,"configured_slot_ticks":12,"over_capacity_slot_ticks":0,"idle_slot_ticks":2,
            "peak_queue_length":0,"peak_in_service":1,
            "admitted_wait_ticks":0,"abandoned_wait_ticks":0,"pending_wait_ticks":0
        })
    );
    let replay = &row["artifact"]["replay_request"];
    assert_eq!(
        replay["payload"]["scenarios"][0]["simulation"]["simulation_options"]["schedule"],
        json!([{"tick":3,"retry":null}])
    );
    assert_eq!(
        run(replay)["result"]["executions"][0]["result"],
        row["result"]
    );

    let mut closed = request.clone();
    closed["payload"]["scenarios"][0]["specification"]["average_degree"] = json!(0);
    let options = &mut closed["payload"]["scenarios"][0]["simulation"]["simulation_options"];
    options["capacity"] = json!(0);
    options["edges"] = json!([]);
    options["schedule"] = json!([{"tick":7,"capacity":2,"service_ticks":2}]);
    options.as_object_mut().unwrap().remove("max_queue_length");
    let response = run(&closed);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([0, 0, 2]), "{response}");
    let replay = &row["artifact"]["replay_request"];
    assert_eq!(
        replay["payload"]["scenarios"][0]["simulation"]["simulation_options"]["capacity"],
        0
    );
    assert_eq!(
        run(replay)["result"]["executions"][0]["result"],
        row["result"]
    );

    let mut invalid = closed.clone();
    invalid["payload"]["scenarios"][0]["simulation"]["simulation_options"]["edges"] =
        json!([[0, 1]]);
    assert_eq!(
        run(&invalid)["error"]["message"],
        "scenario 'retry': average_degree does not match the initial explicit graph"
    );
    invalid["payload"]["scenarios"][0]["specification"]["average_degree"] = json!(-1);
    assert_eq!(run(&invalid)["error"]["code"], "invalid_request");
    invalid["payload"]["scenarios"][0]["specification"]["average_degree"] = json!(0);
    invalid["payload"]["scenarios"][0]["specification"]["network_type"] = json!("ring");
    assert_eq!(run(&invalid)["error"]["code"], "invalid_request");

    invalid["payload"]["scenarios"][0]["specification"]["network_type"] = json!("custom");
    invalid["payload"]["scenarios"][0]["specification"]["average_degree"] = json!(2);
    assert!(
        run(&invalid)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("average_degree cannot exceed num_nodes minus one")
    );

    let mut beyond_horizon = closed.clone();
    beyond_horizon["payload"]["scenarios"][0]["simulation"]["inputs"] = json!({"t":[0,13]});
    assert!(
        run(&beyond_horizon)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("network observation time exceeds the declared scenario horizon")
    );

    let mut artifact = row["artifact"].clone();
    artifact["replay_request"]["payload"]["scenarios"][0]["simulation"]["inputs"]["time"] =
        json!([0, 13]);
    let replay = &artifact["replay_request"];
    assert_eq!(run(replay)["error"]["code"], "invalid_request");
    let import = json!({"schema_version":"2.0","operation":"import_scenario_artifact",
        "payload":{"artifact":artifact}});
    assert_eq!(
        run(&import)["error"]["message"],
        "network observation time exceeds the declared scenario horizon"
    );

    let mut paired = request;
    let mut alternative = closed["payload"]["scenarios"][0].clone();
    alternative["id"] = json!("closed");
    paired["payload"]["scenarios"]
        .as_array_mut()
        .unwrap()
        .push(alternative);
    paired["payload"]["comparisons"] =
        json!([{"baseline":"retry","alternative":"closed","compare_resources":true}]);
    let response = run(&paired);
    let totals = &response["result"]["comparisons"][0]["result"]["resource_differences"];
    assert_eq!(totals["horizon_ticks"], 12);
    assert_eq!(totals["occupied_slot_ticks"], -6);
    assert_eq!(totals["configured_slot_ticks"], -2);
    assert_eq!(totals["idle_slot_ticks"], 4);
    paired["payload"]["scenarios"][1]["simulation"]["simulation_options"]["ticks_per_unit"] =
        json!(2);
    let response = run(&paired);
    assert_eq!(
        response["result"]["comparisons"][0]["error"]["message"],
        "resource comparison requires matching tick resolution and horizon"
    );
}

#[test]
fn historical_logistic_profiles_event_candidates_and_saves_fixed_split() {
    let time: Vec<_> = (0..24).collect();
    let observed: Vec<_> = time
        .iter()
        .map(|&t| {
            let (l, k, x0) = if t < 12 {
                (100., 0.5, 5.)
            } else {
                (200., 0.4, 17.)
            };
            l / (1. + (-k * (f64::from(t) - x0)).exp())
        })
        .collect();
    let mut request = json!({"schema_version":"2.0","operation":"fit_model","model_key":"logistic",
        "payload":{"inputs":{"time":time,"observed":observed},
            "constructor_kwargs":{"event_candidates":[10,12,14]}}});
    let execute = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = execute(&request);
    let result = &response["result"];
    assert_eq!(
        result["state"]["constructor_kwargs"],
        json!({"t_event":12.}),
        "{response}"
    );
    assert!(
        result["event_profile"]["candidates"][1]["objective"]
            .as_f64()
            .unwrap()
            < 1e-10
    );
    let bootstrap = execute(&json!({"schema_version":"2.0","operation":"bootstrap_fit",
        "payload":{"fit_model":"logistic","fit":request["payload"],"draws":3,"seed":42,
            "method":"centered_residual_resampling","forecast_inputs":{"time":[24,25]}}}));
    let forecast = &bootstrap["result"]["forecast"];
    assert_eq!(forecast["succeeded"], 3);
    assert_eq!(forecast["columns"], json!(["prediction"]));
    assert_eq!(forecast["failed"], 0);
    assert!(forecast["pointwise_intervals"][1][0][0].is_number());
    let selection = &bootstrap["result"]["event_selection"];
    assert_eq!(selection["successful_refits"], 3);
    assert_eq!(
        selection["candidates"][1],
        json!({"t_event":12.,"count":3,"frequency":1.})
    );
    let weights: Vec<_> = (0..24).map(|i| if i < 8 { 0 } else { 1 }).collect();
    request["payload"]["fit_options"] = json!({"weights":weights});
    let weighted = execute(&request);
    assert_eq!(
        weighted["result"]["state"]["constructor_kwargs"]["t_event"],
        12.
    );
    assert!(weighted["result"]["event_profile"]["candidates"][0]["error"].is_string());
    assert!(
        weighted["result"]["event_profile"]["candidates"][1]["objective"]
            .as_f64()
            .unwrap()
            < 1e-10
    );
    request["payload"]["constructor_kwargs"]["event_candidates"] = json!([1, 12]);
    assert!(execute(&request)["error"].is_object());
}

#[test]
fn bootstrap_reports_null_forecast_interval_with_one_successful_refit() {
    let request = json!({"schema_version":"2.0","operation":"bootstrap_fit",
        "payload":{"fit_model":"logistic","draws":2,"seed":42,
            "forecast_inputs":{"time":[18]},
            "fit":{"constructor_kwargs":{"event_candidates":[11,12,13]},
                "inputs":{"time":[7,8,9,10,11,12,13,14,15,16,17],
                    "observed":[73.10585786300048,81.85744761936435,87.97970779778824,
                        92.41418199787564,95.35741268224332,23.740584404423508,
                        33.5963229732151,46.39504330019646,61.9051037744775,
                        80.2624679775096,100.1]}}}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(response["error"].is_null(), "{response}");
    let result = &response["result"];
    assert_eq!(result["failure_counts"]["fit_error"], 1);
    assert_eq!(result["forecast"]["succeeded"], 1);
    assert_eq!(result["forecast"]["failed"], 0);
    assert_eq!(result["forecast"]["pointwise_intervals"], json!([[null]]));
}

#[test]
fn adoption_ingestion_rejects_negative_counts_and_rates() {
    for unit in ["count", "rate"] {
        let request = json!({"schema_version":"2.0","operation":"validate_dataset",
            "payload":{"kind":"adoption","time":[0,1],"adoption":[-2,1],"unit":unit}});
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert_eq!(
            response["result"]["failed"],
            json!(["nonnegative:adoption"])
        );
    }
}

#[test]
fn exact_sign_flip_bounds_work_and_counts_zero_ties() {
    let scenarios: Vec<_> = (0..34)
        .map(|id| {
            json!({"id":id.to_string(),"model_key":"logistic",
                "simulation":{"inputs":{"time":[1.]},
                "parameters":{"L":100.,"k":1.,"x0":1.}}})
        })
        .collect();
    let comparison = |count: usize| {
        let pairs: Vec<_> = (0..count)
            .map(|i| json!({"baseline":(2*i).to_string(),"alternative":(2*i+1).to_string()}))
            .collect();
        json!({"pairs":pairs,"interval_mass":0.95,"exact_sign_flip":true})
    };
    let request = json!({"schema_version":"2.0","operation":"execute_scenario_grid",
        "payload":{"scenarios":scenarios,"comparisons":[comparison(16),comparison(17)]}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert_eq!(result["executions"].as_array().unwrap().len(), 34);
    let comparisons = &result["comparisons"];
    let sign_flip = &comparisons[0]["result"]["sign_flip"];
    assert_eq!(sign_flip["assignments"], 65536);
    assert_eq!(sign_flip["extreme_assignments"], 65536);
    assert_eq!(sign_flip["p_value"], 1.);
    assert_eq!(comparisons[1]["error"]["code"], "invalid_comparison");
    assert_eq!(
        comparisons[1]["error"]["message"],
        "exact sign-flip inference supports at most 16 pairs"
    );
}

#[test]
fn execution_discovery_advertises_ordered_benchmark_case_selection() {
    let request =
        json!({"schema_version":"2.0","operation":"discover_execution_modes","payload":{}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert_eq!(
        result["benchmark_case_selection"]["catalog_operation"],
        "discover_benchmark_cases"
    );
    assert_eq!(
        result["benchmark_case_selection"]["explicit_case_ids"],
        true
    );
    assert_eq!(
        result["benchmark_case_selection"]["preserves_case_order"],
        true
    );
    assert_eq!(result["distributed"]["status"], "unqualified");
}

#[test]
fn counterfactual_combines_history_and_event_overrides() {
    let response: serde_json::Value = serde_json::from_str(&innovate_core::execute_json(
        include_str!("fixtures/counterfactual_history_events.json"),
    ))
    .unwrap();
    assert!(response["error"].is_null(), "{response}");
    let result = &response["result"];
    assert_eq!(result["time"], json!([1.0, 2.0, 3.0]));
    // Integrate the piecewise alpha and the linear, then constant, price.
    for (key, integrals) in [
        ("baseline", [0.25, 0.8, 1.4]),
        ("counterfactual", [0.2, 0.4, 0.6]),
    ] {
        for (row, integral) in integrals.into_iter().enumerate() {
            let expected = 1.0 / (1.0 + 9.0 * f64::exp(-integral));
            assert!((result[key][row][0].as_f64().unwrap() - expected).abs() < 1e-8);
        }
    }
    for row in 0..3 {
        assert!(result["difference"][row][1].as_f64().unwrap().abs() < 1e-8);
    }
}

#[test]
fn scenario_comparisons_preserve_order_after_failed_simulation() {
    let response: serde_json::Value = serde_json::from_str(&innovate_core::execute_json(
        include_str!("fixtures/scenario_comparison_isolation.json"),
    ))
    .unwrap();
    assert!(response["error"].is_null(), "{response}");
    let executions = response["result"]["executions"].as_array().unwrap();
    assert_eq!(executions[1]["id"], "bad");
    assert!(!executions[1]["error"].is_null());
    let comparisons = response["result"]["comparisons"].as_array().unwrap();
    assert_eq!(comparisons[0]["alternative"], "bad");
    assert_eq!(comparisons[0]["error"]["code"], "invalid_comparison");
    assert!(comparisons[0]["result"].is_null());
    for (row, places, effect) in [
        (&comparisons[1], [2, 1], 50.),
        (&comparisons[2], [1, 2], -50.),
    ] {
        assert!(row["error"].is_null());
        let result = &row["result"];
        assert_eq!(result["ranking"]["baseline_place"], places[0]);
        assert_eq!(result["ranking"]["alternative_place"], places[1]);
        assert_eq!(result["difference"][0], effect);
    }
    assert_eq!(
        comparisons[1]["result"]["relative_margin_bounds"]["baseline"],
        comparisons[2]["result"]["relative_margin_bounds"]["alternative"]
    );
}

#[test]
fn resource_comparison_requires_resource_outputs_from_both_scenarios() {
    let mut payload: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/kairos_resource_comparisons.json")).unwrap();
    payload["scenarios"][2] = json!({"id":"bad","model_key":"logistic",
        "simulation":{"inputs":{"time":[0,6,7,9,11]},"parameters":{"L":100,"k":1,"x0":1}}});
    payload["comparisons"]
        .as_array_mut()
        .unwrap()
        .push(json!({"baseline":"base","alternative":"bad"}));
    let mut permanently_closed = payload["scenarios"][1].clone();
    permanently_closed["id"] = json!("never_open");
    permanently_closed["simulation"]["simulation_options"]["schedule"] = json!([]);
    permanently_closed["simulation"]["simulation_options"]["max_queue_length"] = json!(1);
    permanently_closed["simulation"]["simulation_options"]["max_wait_ticks"] = json!(5);
    payload["scenarios"]
        .as_array_mut()
        .unwrap()
        .push(permanently_closed);
    payload["comparisons"]
        .as_array_mut()
        .unwrap()
        .push(json!({"baseline":"base","alternative":"never_open","compare_resources":true}));
    let mut retry = payload["scenarios"][3].clone();
    retry["id"] = json!("retry_closed");
    retry["simulation"]["simulation_options"]["max_queue_length"] = json!(0);
    retry["simulation"]["simulation_options"]["retry"] = json!({"delay_ticks":1,"max_retries":1});
    payload["scenarios"].as_array_mut().unwrap().push(retry);
    payload["comparisons"]
        .as_array_mut()
        .unwrap()
        .push(json!({"baseline":"base","alternative":"retry_closed","compare_resources":true}));
    for (index, id) in [(0, "base_repeat"), (1, "closed_repeat")] {
        let mut repeat = payload["scenarios"][index].clone();
        repeat["id"] = json!(id);
        repeat["simulation"]["simulation_options"]["seed"] = json!(43);
        payload["scenarios"].as_array_mut().unwrap().push(repeat);
    }
    let discovery: serde_json::Value = serde_json::from_str(&innovate_core::execute_json(
        r#"{"schema_version":"2.0","operation":"discover_scenarios","payload":{"key":"network"}}"#,
    ))
    .unwrap();
    let metrics = discovery["result"]["scenarios"][0]["resource_comparison"]["ensemble_metrics"]
        .as_array()
        .unwrap();
    for metric in metrics {
        payload["comparisons"].as_array_mut().unwrap().push(json!({
            "pairs":[{"baseline":"base","alternative":"closed"},
                     {"baseline":"base_repeat","alternative":"closed_repeat"}],
            "require_distinct_pair_seeds":true,"interval_mass":0.9,
            "evaluation_metric":metric
        }));
    }
    let request =
        json!({"schema_version":"2.0","operation":"execute_scenario_grid","payload":payload});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let result = &response["result"];
    assert!(result["executions"][2]["error"].is_null(), "{response}");
    let rows = result["comparisons"].as_array().unwrap();
    let ensembles = &rows[5..];
    assert_eq!(ensembles.len(), metrics.len());
    for (metric, row) in metrics.iter().zip(ensembles) {
        let key = metric.as_str().unwrap().strip_prefix("resource.").unwrap();
        let ensemble = &row["result"];
        let effect = rows[0]["result"]["resource_differences"][key]
            .as_i64()
            .unwrap() as f64;
        assert_eq!(ensemble["resource_metric"], key, "{row}");
        assert_eq!(ensemble["differences"], json!([effect, effect]), "{key}");
        assert_eq!(ensemble["mean_standard_error"], 0., "{key}");
    }
    assert_eq!(
        rows[0]["result"]["resource_differences"]["configured_slot_ticks"],
        -3
    );
    assert_eq!(
        rows[0]["result"]["resource_differences"]["peak_in_service"],
        1
    );
    assert_eq!(
        rows[1]["error"]["message"],
        "resource comparison requires two Kairos network resource results"
    );
    assert!(rows[1]["result"].is_null());
    assert!(rows[2]["error"].is_null());
    assert!(rows[2]["result"].get("resource_differences").is_none());
    let closed = &rows[3]["result"]["resource_differences"];
    assert_eq!(closed["completed_services"], -3);
    assert_eq!(closed["balked_requests"], 2);
    assert_eq!(closed["abandoned_requests"], 1);
    assert_eq!(closed["abandoned_wait_ticks"], 5);
    assert_eq!(closed["waiting_at_horizon"], 0);
    assert_eq!(closed["in_service_at_horizon"], 0);
    let retried = &rows[4]["result"]["resource_differences"];
    assert_eq!(retried["balked_requests"], 6);
    assert_eq!(retried["abandoned_requests"], 0);

    let mut short = request;
    for scenario in short["payload"]["scenarios"].as_array_mut().unwrap() {
        scenario["simulation"]["inputs"]["time"] = json!([0, 7]);
    }
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&short.to_string())).unwrap();
    let reopening = &response["result"]["comparisons"][0]["result"]["resource_differences"];
    assert_eq!(reopening["waiting_at_horizon"], 1);
    assert_eq!(reopening["in_service_at_horizon"], 2);
}

#[test]
fn edgeless_dataset_ingests_and_replays() {
    let dataset = json!({"kind":"network_edges","source":[],"target":[],"weight":[],"directed":false,
        "provenance":{"source":"synthetic","license":"CC-BY-4.0","extraction_time":"2026-11-07T00:00:00Z"}});
    let run = |payload| {
        let request =
            json!({"schema_version":"2.0","operation":"ingest_dataset","payload":payload});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let response = run(json!({"dataset":dataset}));
    assert!(response["error"].is_null(), "{response}");
    let artifact = &response["result"];
    assert_eq!(artifact["dataset"]["source"], json!([]));
    assert_eq!(run(json!({"artifact":artifact}))["result"], *artifact);
    let mut misaligned = dataset;
    misaligned["target"] = json!(["a"]);
    assert_eq!(
        run(json!({"dataset":misaligned}))["error"]["validation"]["failed"][0],
        "column_alignment"
    );
}

#[test]
fn dataset_ingestion_defaults_match_explicit_values() {
    let run = |mut dataset: serde_json::Value| {
        if dataset.get("provenance").is_none() {
            dataset["provenance"] = json!({"source":"synthetic","license":"CC-BY-4.0",
                "extraction_time":"2026-11-07T00:00:00Z"});
        }
        let request = json!({"schema_version":"2.0","operation":"ingest_dataset","payload":{"dataset":dataset}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    for (mut dataset, key, values) in [
        (
            json!({"kind":"policy_timing","event_times":[0,2],"event_effects":[0,-0.5]}),
            "event_labels",
            json!(["event_0", "event_1"]),
        ),
        (
            json!({"kind":"network_edges","source":["a"],"target":["b"]}),
            "weight",
            json!([1.0]),
        ),
    ] {
        let implicit = run(dataset.clone());
        assert!(implicit["error"].is_null(), "{implicit}");
        assert_eq!(implicit["result"]["dataset"][key], values);
        dataset[key] = values;
        assert_eq!(run(dataset), implicit);
        let mut artifact = implicit["result"].clone();
        artifact["dataset"].as_object_mut().unwrap().remove(key);
        assert_eq!(
            run(artifact["dataset"].clone())["error"]["code"],
            "invalid_request"
        );
        let request = json!({"schema_version":"2.0","operation":"ingest_dataset","payload":{"artifact":artifact}});
        let response: serde_json::Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert_eq!(
            response["error"]["message"],
            "dataset content checksum does not match provenance"
        );
    }
}

#[test]
fn kairos_scenario_artifact_replays_continuous_hazard_wakeup() {
    let request = serde_json::from_str(include_str!("fixtures/kairos_hazard_replay.json")).unwrap();
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = run(&request);
    let row = &response["result"]["executions"][0];
    assert_eq!(row["result"]["counts"], json!([0, 0, 0, 1]), "{response}");
    assert_eq!(row["result"]["requests"], json!([{"tick":3,"agent":0}]));
    assert_eq!(
        run(&row["artifact"]["replay_request"])["result"]["executions"][0]["result"],
        row["result"]
    );
    let mut import = json!({"schema_version":"2.0","operation":"import_scenario_artifact",
        "payload":{"artifact":row["artifact"]}});
    let market = &mut import["payload"]["artifact"]["replay_request"]["payload"]["scenarios"][0]["simulation"]
        ["parameters"]["m"];
    assert_eq!(*market, json!(1.0));
    *market = json!(1);
    assert_eq!(run(&import)["result"]["replay_available"], true);
    import["payload"]["artifact"]["replay_request"]["payload"]["scenarios"][0]["simulation"]["parameters"]
        ["m"] = json!(2);
    assert_eq!(
        run(&import)["error"]["message"],
        "submitted simulation does not resolve to the embedded replay payload"
    );
}

#[test]
fn scenario_grid_paused_network_preserves_seed_and_resumes() {
    let mut request: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/kairos_hazard_replay.json")).unwrap();
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let expected = run(&request);
    let mut other = request["payload"]["scenarios"][0].clone();
    other["id"] = json!("other");
    request["payload"]["scenarios"]
        .as_array_mut()
        .unwrap()
        .push(other);
    request["payload"]["rank_scenarios"] = json!(["wake", "other"]);
    let options = "/payload/scenarios/0/simulation/simulation_options";
    request.pointer_mut(options).unwrap()["pause_at_tick"] = json!(2);
    let paused = run(&request);
    let row = &paused["result"]["executions"][0];
    assert_eq!(row["result"]["status"], "paused", "{paused}");
    assert_eq!(row["artifact"]["seed"], 42);
    assert_eq!(
        paused["result"]["ranking"]["error"]["message"],
        "comparison requires a completed simulation; resume the paused scenario"
    );
    let mut resume = row["artifact"]["replay_request"].clone();
    assert_eq!(
        run(&resume)["result"]["executions"][0]["result"],
        row["result"]
    );
    let resume_options = resume
        .pointer_mut(options)
        .unwrap()
        .as_object_mut()
        .unwrap();
    resume_options.remove("pause_at_tick");
    resume_options.insert(
        "resume_checkpoint".into(),
        row["result"]["checkpoint"].clone(),
    );
    assert_eq!(
        run(&resume)["result"]["executions"][0]["result"],
        expected["result"]["executions"][0]["result"]
    );
}

#[test]
fn empirical_intervals_keep_legacy_width_and_reject_invalid_inputs() {
    let mut request = json!({"schema_version":"2.0","operation":"calibrate_prediction_intervals",
        "payload":{"time":[1,2,3],"observed":[11,18,35],"predicted":[10,20,30],
            "confidence":0.5,"holdout":[0,1,0]}});
    let execute = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = execute(&request);
    assert_eq!(response["result"]["lower"], json!([8., 18., 28.]));
    assert_eq!(response["result"]["upper"], json!([12., 22., 32.]));
    assert_eq!(response["result"]["diagnostics"]["holdout_coverage"], 1.0);
    request["payload"]["confidence"] = json!(1.5);
    assert_eq!(execute(&request)["error"]["code"], "invalid_request");
    request["payload"]["confidence"] = json!(0.5);
    request["payload"]["predicted"] = json!([10, 20]);
    assert_eq!(execute(&request)["error"]["code"], "invalid_request");
    request["payload"]["predicted"] = json!([10, 20, 30]);
    for (field, invalid, valid) in [
        ("observed", json!([11, 18]), json!([11, 18, 35])),
        ("holdout", json!([0, 1]), json!([0, 1, 0])),
        ("confidence", json!(-0.5), json!(0.5)),
    ] {
        request["payload"][field] = invalid;
        assert_eq!(
            execute(&request)["error"]["code"],
            "invalid_request",
            "{field}"
        );
        request["payload"][field] = valid;
    }
    request["payload"]["time"] = json!([1, 1, 3]);
    assert_eq!(execute(&request)["error"]["code"], "invalid_request");
    request["payload"] = json!({"time":[1,2],"observed":[f64::MAX,-f64::MAX],"predicted":[0,0]});
    let result = execute(&request);
    assert_eq!(result["result"]["diagnostics"]["residual_rmse"], f64::MAX);
    assert_eq!(result["result"]["diagnostics"]["residual_mean"], 0.0);
    request["payload"]["observed"] = json!([f64::MAX, 0]);
    request["payload"]["predicted"] = json!([-f64::MAX, 0]);
    assert_eq!(execute(&request)["error"]["code"], "invalid_request");
}

#[test]
fn split_conformal_intervals_exclude_evaluation_residuals() {
    let mut request = json!({"schema_version":"2.0","operation":"calibrate_prediction_intervals",
        "payload":{"time":[1,2,3,4],"observed":[11,12,13,100],"predicted":[10,10,10,10],
            "confidence":0.5,"holdout":[0,0,0,1],"method":"split_conformal"}});
    let execute = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let response = execute(&request);
    assert_eq!(response["result"]["metadata"]["interval_half_width"], 2.0);
    assert_eq!(response["result"]["diagnostics"]["holdout_coverage"], 0.0);
    request["payload"]["observed"][3] = json!(11);
    let response = execute(&request);
    assert_eq!(response["result"]["metadata"]["interval_half_width"], 2.0);
    assert_eq!(response["result"]["diagnostics"]["holdout_coverage"], 1.0);
    request["payload"]["confidence"] = json!(0.9);
    assert_eq!(execute(&request)["error"]["code"], "invalid_request");
    // Each distinct score is equally likely to be the held-out score.
    request["payload"]["observed"] = json!([11, 12, 13, 14]);
    for (confidence, expected_covered) in [(0.25, 1), (0.5, 2), (0.75, 3)] {
        request["payload"]["confidence"] = json!(confidence);
        let mut covered = 0;
        for held_out in 0..4 {
            let mut mask = [0; 4];
            mask[held_out] = 1;
            request["payload"]["holdout"] = json!(mask);
            let result = execute(&request);
            assert!(result["error"].is_null());
            covered += usize::from(result["result"]["diagnostics"]["holdout_coverage"] == 1.0);
        }
        assert_eq!(covered, expected_covered);
    }
}

#[test]
fn kernel_operation_discovery_lists_unique_native_routes() {
    let request = json!({"schema_version":"2.0","operation":"list_kernel_operations"});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(response["error"].is_null());
    let operations = response["result"]["operations"].as_array().unwrap();
    let names: std::collections::HashSet<_> =
        operations.iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names.len(), operations.len());
    for name in [
        "fit_model",
        "list_kernel_operations",
        "execute_scenario_grid",
        "calibrate_prediction_intervals",
    ] {
        assert!(names.contains(name));
    }
}

#[test]
fn policy_scenario_compares_supplied_paths_and_keeps_metadata() {
    let mut payload = json!({"method":"policy_scenario","time":[1,2,3],
        "baseline":[10,20,30],"intervention":[10,25,45],"observed":[10,25,45],
        "scenario_name":"test_scenario","assumptions":["a1"],"covariates":{"cov1":[0.1,0.2,0.3]}});
    let run = |payload: &serde_json::Value| -> serde_json::Value {
        let request =
            json!({"schema_version":"2.0","operation":"summarize_causal","payload":payload});
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let result = run(&payload);
    assert_eq!(result["result"]["metadata"]["incremental_effect"], 20.0);
    assert_eq!(result["result"]["metadata"]["relative_lift_final"], 0.5);
    assert_eq!(
        result["result"]["metadata"]["covariates"],
        payload["covariates"]
    );
    assert_eq!(result["result"]["diagnostics"]["intervention_rmse"], 0.0);
    payload["baseline"][2] = json!(0);
    assert!(run(&payload)["result"]["metadata"]["relative_lift_final"].is_null());
    payload["baseline"] = json!([10, 20]);
    assert_eq!(run(&payload)["error"]["code"], "invalid_request");
}

#[test]
fn regime_ensemble_normalizes_weights_and_validates_alignment() {
    let mut request = json!({"schema_version":"2.0","operation":"compose_regime_ensemble",
        "payload":{"time":[1,2],"predictions":{"a":[1,2],"b":[3,4]},"weights":{"a":1,"b":3},
            "observed":[2.5,3.5],"assumptions":["shared time grid"]}});
    let run = |request: &serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
    };
    let result = run(&request);
    assert_eq!(result["result"]["mean"], json!([2.5, 3.5]));
    assert_eq!(result["result"]["diagnostics"]["rmse"], 0.);
    assert_eq!(
        result["result"]["metadata"]["assumptions"],
        json!(["shared time grid"])
    );
    request["payload"]["weights"] = serde_json::Value::Null;
    assert_eq!(run(&request)["result"]["mean"], json!([2., 3.]));
    for weights in [json!({"a":1,"c":1}), json!({"a":0,"b":0})] {
        request["payload"]["weights"] = weights;
        assert_eq!(run(&request)["error"]["code"], "invalid_request");
    }
}

#[test]
fn diagnostics_artifacts_preserve_columns_and_missing_values() {
    let request = json!({"schema_version":"2.0","operation":"diagnostics_artifact_tables","payload":{
    "schema_version":"1.0","model_name":"bass","support_level":"supported","artifacts":{
        "ordered":{"kind":"residuals","columns":["z","a"],"rows":[{"a":1,"z":2},{"a":3},false]},
        "single":{"columns":["value"],"rows":[{"value":4},{}]},
        "summary":{"value":2}
    }}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    let tables = &response["result"]["tables"];
    assert_eq!(tables["ordered"]["columns"], json!(["z", "a"]));
    assert_eq!(
        tables["ordered"]["rows"],
        json!([[2, 1], [null, 3], [null, null]])
    );
    assert_eq!(tables["single"]["rows"], json!([[4], [null]]));
    assert_eq!(tables["ordered"]["metadata"]["model_name"], "bass");
    assert!(tables.get("summary").is_none());
}

#[test]
fn advanced_capability_discovery_reports_ordered_native_routes() {
    let run = |operation| {
        let request = json!({"schema_version":"2.0","operation":operation});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let result = &run("list_advanced_capabilities")["result"];
    let entries = result["capabilities"].as_array().unwrap();
    let keys: Vec<_> = entries
        .iter()
        .map(|item| item["key"].as_str().unwrap())
        .collect();
    assert_eq!(
        keys,
        [
            "policy_scenario",
            "regime_ensemble",
            "streaming_update",
            "uncertainty_calibration",
            "competition_interaction_prediction",
            "competition_interaction_fitting",
            "competition_cross_elasticity",
            "competition_equilibrium",
        ]
    );
    let operations = run("list_kernel_operations");
    for entry in entries {
        assert!(
            operations["result"]["operations"]
                .as_array()
                .unwrap()
                .contains(&entry["operation"])
        );
        assert_eq!(entry["supported_backends"], json!(["rust_native"]));
        assert_eq!(entry["optional_dependencies"], json!([]));
    }
    assert_eq!(entries[1]["stability"], "provisional");
    assert_eq!(entries[2]["supports_incremental"], true);
    assert_eq!(result["pending"], json!([]));
}

#[test]
fn streaming_update_appends_cumulative_observations() {
    let mut request = json!({"schema_version":"2.0","operation":"update_streaming_forecast","payload":{
        "previous_time":[1,2],"previous_observed":[2,4],"new_time":[3],"new_observed":[6]}});
    let run = |request: &serde_json::Value| {
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let result = run(&request);
    assert_eq!(result["result"]["mean"], json!([2., 4., 6.]));
    assert_eq!(result["result"]["metadata"]["state"]["total_count"], 3);
    assert_eq!(result["result"]["diagnostics"]["growth_rate"], 0.5);
    request["payload"]["new_observed"] = json!([3]);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
    request["payload"]["new_observed"] = json!([6]);
    request["payload"]["new_time"] = json!([1]);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn residual_diagnostics_return_analytic_acf_and_pacf() {
    let mut request = json!({"schema_version":"2.0","operation":"residual_diagnostic_series","payload":{
        "time":[1,2,3,4],"residuals":[-1,1,-1,1],"lags":2}});
    let run = |request: &serde_json::Value| {
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let result = run(&request);
    assert_eq!(result["result"]["acf"], json!([1., -0.75, 0.5]));
    assert!((result["result"]["pacf"][2].as_f64().unwrap() + 1. / 7.).abs() < 1e-12);
    request["payload"]["residuals"] = json!([1, 1, 1, 1]);
    assert_eq!(run(&request)["result"]["pacf"], json!([1., null, null]));
    request["payload"]["lags"] = json!(1025);
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn residual_svg_contains_configured_panels_and_escapes_title() {
    let mut request = json!({"schema_version":"2.0","operation":"render_residual_diagnostics_svg","payload":{
        "time":[1,2,3,4],"residuals":[-1,1,-1,1],"config":{"title":"A < B & C","lags":2,"figsize":[8,4],"color_residuals":"#123456","color_acf":"C3"}}});
    let run = |request: &serde_json::Value| {
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    let result = run(&request);
    let svg = result["result"]["svg"].as_str().unwrap();
    assert!(svg.starts_with("<svg"));
    assert!(svg.contains("A &lt; B &amp; C"));
    assert!(svg.contains("Partial autocorrelation"));
    assert_eq!(svg.matches("<circle").count(), 4);
    assert!(svg.contains(r#"width="768" height="384""#));
    assert!(svg.contains(r##"fill="#123456""##));
    assert!(svg.contains(r##"stroke="#d62728""##));
    assert_eq!(result["result"]["series"]["lags"], json!([0, 1, 2]));
    request["payload"]["config"]["acf_only"] = json!(true);
    assert!(
        !run(&request)["result"]["svg"]
            .as_str()
            .unwrap()
            .contains("Partial autocorrelation")
    );
}

#[test]
fn oecd_request_preserves_dimensions_and_date_defaults() {
    let mut request = json!({"schema_version":"2.0","operation":"build_oecd_dataset_request","payload":{
        "dataset_id":"OECD.SDD.STES,DSD_STES@DF_CLI","dimension_key":".M.LI...AA...H"}});
    let run = |request: &serde_json::Value| {
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    assert_eq!(
        run(&request)["result"]["url"],
        "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI/.M.LI...AA...H?startPeriod=1960&endPeriod=2020&dimensionAtObservation=AllDimensions&format=csvfile"
    );
    request["payload"]["start_date"] = json!("2023-02");
    request["payload"]["end_date"] = json!("2023-03");
    assert!(
        run(&request)["result"]["url"]
            .as_str()
            .unwrap()
            .contains("startPeriod=2023-02&endPeriod=2023-03")
    );
    request["payload"]["dimension_key"] = json!("?injected=true");
    assert_eq!(run(&request)["error"]["code"], "invalid_request");
}

#[test]
fn oecd_fetch_rejects_invalid_requests_before_network_access() {
    let request = json!({"schema_version":"2.0","operation":"fetch_oecd_dataset","payload":{
        "dataset_id":"OECD.SDD.STES,DSD_STES@DF_CLI","dimension_key":"?injected=true"}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert_eq!(response["error"]["code"], "invalid_request");
    assert!(response["result"].is_null());
}

#[test]
#[ignore = "requires the live public OECD SDMX endpoint"]
fn oecd_fetch_returns_csv_from_live_endpoint() {
    let request = json!({"schema_version":"2.0","operation":"fetch_oecd_dataset","payload":{
        "dataset_id":"OECD.SDD.STES,DSD_STES@DF_CLI","dimension_key":".M.LI...AA...H",
        "start_date":"2023-02","end_date":"2023-03"}});
    let response: serde_json::Value =
        serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
    assert!(response["error"].is_null(), "{}", response["error"]);
    let csv = response["result"]["csv"].as_str().unwrap();
    let mut lines = csv.lines();
    assert!(lines.next().unwrap().contains("TIME_PERIOD,OBS_VALUE"));
    assert!(lines.any(|row| row.contains(",2023-02,") || row.contains(",2023-03,")));
}

#[test]
fn policy_document_import_bounds_and_validates_json() {
    let document = json!({
        "intervention":{"name":"Pilot","timing":"post","comparator":"control"},
        "causal_model":{"name":"Evaluation","treatment_variable":"policy",
            "outcome_variable":"adoption","confounders":["age"]}
    })
    .to_string();
    let run = |document: serde_json::Value| {
        let request = json!({"schema_version":"2.0","operation":"summarize_causal",
            "payload":{"method":"import_policy_specification","document":document}});
        serde_json::from_str::<serde_json::Value>(&innovate_core::execute_json(
            &request.to_string(),
        ))
        .unwrap()
    };
    assert_eq!(
        run(json!(document))["result"]["intervention"]["name"],
        "Pilot"
    );
    for (document, message) in [
        (
            json!({}),
            "policy specification document must be a JSON string",
        ),
        (json!("{"), "invalid policy specification JSON"),
        (
            json!("é".repeat(524_289)),
            "policy specification document exceeds 1 MiB",
        ),
    ] {
        assert_eq!(run(document)["error"]["message"], message);
    }
    assert_eq!(run(json!("[]"))["error"]["code"], "invalid_request");
    let mut specification: serde_json::Value = serde_json::from_str(&document).unwrap();
    specification["__class__"] = json!("unrecognized");
    assert_eq!(
        run(json!(specification.to_string()))["error"]["message"],
        "policy specification document contains unknown fields"
    );
    specification.as_object_mut().unwrap().remove("__class__");
    for section in ["intervention", "causal_model"] {
        specification[section]["__class__"] = json!("unrecognized");
        assert_eq!(
            run(json!(specification.to_string()))["error"]["message"],
            "policy specification contains unknown contract fields"
        );
        specification[section]
            .as_object_mut()
            .unwrap()
            .remove("__class__");
    }
    specification["intervention"]["timing"] = json!(123);
    assert_eq!(
        run(json!(specification.to_string()))["error"]["message"],
        "invalid intervention contract fields"
    );
    let mut specification: serde_json::Value = serde_json::from_str(&document).unwrap();
    specification["intervention"]["timing"] = json!("post");
    specification["intervention"]["start_time"] = json!("2020");
    assert_eq!(
        run(json!(specification.to_string()))["error"]["message"],
        "invalid intervention contract fields"
    );
    specification["intervention"]["start_time"] = json!(2020);
    specification["causal_model"]["confounders"] = json!([7]);
    assert_eq!(
        run(json!(specification.to_string()))["error"]["message"],
        "invalid causal model contract fields"
    );
}
