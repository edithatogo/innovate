//! Track 5 public-operation preservation; independent dependency fixtures are separate.
use serde_json::{Value, json};

fn execute(operation: &str, payload: Value) -> Value {
    let request = json!({"schema_version":"2.0", "operation":operation, "payload":payload});
    serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
}

fn replay_cases() -> Vec<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/statrs_20261115/behavior-baseline.json"
    ))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["provenance"]["base_commit"],
        "42189a223fd44c823ca11aec689ea67b11c25790"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    cases.clone()
}

#[test]
fn statrs_migration_keeps_seeded_draws_acceptance_and_grid_mass() {
    let mut checked = 0;
    for case in replay_cases() {
        if case["request"]["operation"] == "bootstrap_fit" {
            continue;
        }
        let response: Value =
            serde_json::from_str(&innovate_core::execute_json(&case["request"].to_string()))
                .unwrap();
        assert_eq!(response, case["expected"], "{}", case["id"]);
        checked += 1;
    }
    assert_eq!(checked, 3);
}

#[test]
fn statrs_migration_bootstrap_keeps_empirical_type_seven_intervals() {
    let mut checked = 0;
    for case in replay_cases() {
        if case["request"]["operation"] != "bootstrap_fit" {
            continue;
        }
        let response: Value =
            serde_json::from_str(&innovate_core::execute_json(&case["request"].to_string()))
                .unwrap();
        assert!(response["error"].is_null(), "{response}");
        let result = &response["result"];
        assert_eq!(result["succeeded"], 8);
        assert_eq!(result["failed"], 0);
        assert_eq!(result["interval_method"], "percentile");
        assert_eq!(result["interval_mass"], 0.8);
        assert_eq!(result["method"], case["expected"]["result"]["method"]);
        // For eight order statistics, the 10th and 90th type-7 percentiles
        // are 0.3*x[0] + 0.7*x[1] and 0.7*x[6] + 0.3*x[7]. These equations
        // qualify interval semantics independently of the owned fit solver.
        for (name, values) in result["estimates"].as_object().unwrap() {
            let mut ordered: Vec<f64> = serde_json::from_value(values.clone()).unwrap();
            assert_eq!(ordered.len(), 8);
            ordered.sort_by(f64::total_cmp);
            let expected = [
                0.3 * ordered[0] + 0.7 * ordered[1],
                0.7 * ordered[6] + 0.3 * ordered[7],
            ];
            for (index, expected) in expected.into_iter().enumerate() {
                let actual = result["parameter_intervals"][name][index].as_f64().unwrap();
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{} {name}: {actual} != {expected}",
                    case["id"]
                );
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 2);
}

#[test]
fn statrs_migration_antithetic_ess_has_the_analytic_cap() {
    // Identical alternating chains have negative lag-one autocorrelation. The
    // declared Geyer floor is 1/log10(N), giving ESS = N*log10(N), even above N.
    let chains = json!([[0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1]]);
    let response = execute("bayesian_effective_sample_size", json!({"chains":chains}));
    assert!(response["error"].is_null(), "{response}");
    let result = &response["result"];
    assert_eq!(result["method"], "split_initial_monotone_sequence");
    assert_eq!(result["draws_used"], 16);
    assert_eq!(result["draws_per_split"], 4);
    let expected = 16. * 16_f64.log10();
    for key in ["bulk_ess", "lower_tail_ess"] {
        assert!(
            (result[key].as_f64().unwrap() - expected).abs() < 1e-12,
            "{key}: {response}"
        );
    }
    // With ties at the maximum, the 95th-percentile indicator is constantly
    // one. Its variance is zero, so upper and combined tail ESS remain null.
    assert!(result["upper_tail_ess"].is_null());
    assert!(result["tail_ess"].is_null());
}

#[test]
fn statrs_migration_odd_chain_middle_is_excluded_before_ranking() {
    let even = json!([
        [-2, 0, 0, 2, -2, 0, 0, 2],
        [-20, -10, 10, 20, -20, -10, 10, 20]
    ]);
    let odd = json!([
        [-2, 0, 0, 2, 1e200, -2, 0, 0, 2],
        [-20, -10, 10, 20, -1e200, -20, -10, 10, 20]
    ]);
    for operation in ["bayesian_split_rhat", "bayesian_effective_sample_size"] {
        let mut left = execute(
            operation,
            json!({"method":"rank_normalized_split_rhat", "chains":even}),
        );
        let mut right = execute(
            operation,
            json!({"method":"rank_normalized_split_rhat", "chains":odd}),
        );
        assert!(left["error"].is_null(), "{left}");
        assert!(right["error"].is_null(), "{right}");
        // Draw-count disclosure differs; the diagnosed values must not.
        left["result"]
            .as_object_mut()
            .unwrap()
            .remove("draws_per_chain");
        right["result"]
            .as_object_mut()
            .unwrap()
            .remove("draws_per_chain");
        assert_eq!(left, right);
    }
}

#[test]
fn statrs_migration_public_diagnostics_reject_invalid_domains() {
    let cases = [
        (
            json!({"chains":[[0,1,2,3]]}),
            "split R-hat requires 2 to 16 chains",
        ),
        (
            json!({"chains":[[0,1,2],[0,1,2]]}),
            "split R-hat requires 4 to 10000 draws per chain",
        ),
        (
            json!({"chains":[[0,1,2,3],[0,1,2,3,4]]}),
            "chains must have equal draw counts",
        ),
        (
            json!({"chains":[[0,1,null,3],[0,1,2,3]]}),
            "chain draws must be finite numbers",
        ),
        (
            json!({"chains":[[0,1,"NaN",3],[0,1,2,3]]}),
            "chain draws must be finite numbers",
        ),
        (
            json!({"method":"normal_quantile", "chains":[[0,1,2,3],[0,1,2,3]]}),
            "unknown split R-hat method",
        ),
    ];
    for (payload, message) in cases {
        let response = execute("bayesian_split_rhat", payload);
        assert_eq!(response["error"]["message"], message, "{response}");
        assert!(response["result"].is_null());
    }
    let response = execute(
        "bayesian_effective_sample_size",
        json!({"chains":[[0,1,2,3],[0,1,2,3]]}),
    );
    assert_eq!(
        response["error"]["message"],
        "effective sample size requires at least six draws per chain"
    );
}

#[test]
fn statrs_migration_stuck_diagnostics_preserve_nulls_and_counts() {
    let chains = json!([[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]);
    let rhat = execute(
        "bayesian_split_rhat",
        json!({"method":"rank_normalized_split_rhat", "chains":chains}),
    );
    assert!(rhat["error"].is_null(), "{rhat}");
    assert_eq!(rhat["result"]["status"], "zero_within_split_variance");
    for key in ["rhat", "bulk_rhat", "folded_rhat"] {
        assert!(rhat["result"][key].is_null(), "{key}: {rhat}");
    }
    let ess = execute("bayesian_effective_sample_size", json!({"chains":chains}));
    assert!(ess["error"].is_null(), "{ess}");
    assert_eq!(ess["result"]["draws_used"], 12);
    for key in ["bulk_ess", "lower_tail_ess", "upper_tail_ess", "tail_ess"] {
        assert!(ess["result"][key].is_null(), "{key}: {ess}");
    }
}
