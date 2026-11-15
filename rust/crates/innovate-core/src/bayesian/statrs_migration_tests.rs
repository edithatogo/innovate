//! Independent characterization before and after the Track 5 quantile migration.
//!
//! These tests intentionally need no statrs import: they first execute against
//! the original implementation. Dependency-only CDF/PDF qualification is separate.

use super::{GaussianPriors, log_likelihood, normal_quantile, observations, rank_scores};
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/statrs_20261115/normal.json"
    ))
    .expect("checked-in independent normal reference")
}

fn assert_reference(actual: f64, case: &Value) {
    let id = case["id"].as_str().expect("reference case ID");
    match case["expected"].as_str() {
        Some("-inf") => assert_eq!(actual, f64::NEG_INFINITY, "{id}"),
        Some("inf") => assert_eq!(actual, f64::INFINITY, "{id}"),
        Some("nan") => assert!(actual.is_nan(), "{id}: expected NaN, got {actual}"),
        Some(other) => panic!("{id}: unknown reference classification {other}"),
        None => {
            let expected = case["expected"].as_f64().expect("numeric reference");
            let tolerance = case["abs_tol"].as_f64().expect("absolute tolerance")
                + case["rel_tol"].as_f64().unwrap_or(0.) * expected.abs();
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "{id}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.17e}"
            );
        }
    }
}

#[test]
fn quantiles_match_independent_center_tail_and_attainable_domain_references() {
    let data = fixture();
    let cases: Vec<_> = data["normal_quantile"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["legacy_domain"] == true)
        .collect();
    assert_eq!(cases.len(), 13, "all predeclared legacy-domain cases run");
    for case in cases {
        assert_reference(normal_quantile(case["p"].as_f64().unwrap()), case);
    }
}

#[test]
fn maximum_pooled_rank_count_stays_inside_the_qualified_probability_domain() {
    // The public limit is sixteen chains with ten thousand draws each. Both
    // halves are retained for an even draw count, hence 160,000 pooled ranks.
    let values: Vec<_> = (0..160_000).map(f64::from).collect();
    let scores = rank_scores(&values);
    assert_eq!(scores.len(), 160_000);
    assert!(scores.iter().all(|score| score.is_finite()));
    assert!(scores.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(scores[0] > -8. && scores[scores.len() - 1] < 8.);

    let data = fixture();
    for (id, actual) in [
        ("smallest_reachable_blom", scores[0]),
        ("largest_reachable_blom", scores[scores.len() - 1]),
    ] {
        let case = data["normal_quantile"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["id"] == id)
            .expect("maximum-count endpoint reference");
        assert_reference(actual, case);
    }
}

#[test]
fn tied_rank_scores_keep_average_ranks_and_signed_zero_equivalence() {
    let values = [9., -0., 2., 0., -3., 9., 0., 2.];
    // Average ranks are 1, 3, 5.5 and 7.5. Their independently evaluated
    // mpmath normal quantiles at p=(rank-3/8)/(8+1/4) are listed explicitly.
    let expected = [
        1.0968035620935133,
        -0.47278912099226733,
        0.308665805694934,
        -0.47278912099226733,
        -1.4342001596863794,
        1.0968035620935133,
        -0.47278912099226733,
        0.308665805694934,
    ];
    let scores = rank_scores(&values);
    for (index, (actual, reference)) in scores.iter().zip(expected).enumerate() {
        assert!(
            (actual - reference).abs() <= 2e-10,
            "rank {index}: actual={actual}, reference={reference}"
        );
    }
    assert_eq!(scores[1], scores[3]);
    assert_eq!(scores[1], scores[6]);
    assert_eq!(scores[0], scores[5]);
    assert_eq!(scores[2], scores[7]);

    let reversed: Vec<_> = values.into_iter().rev().collect();
    assert_eq!(
        rank_scores(&reversed),
        scores.into_iter().rev().collect::<Vec<_>>()
    );
}

#[test]
fn parsed_gaussian_priors_preserve_unnormalised_scores_and_extreme_scales() {
    let data = fixture();
    let cases: Vec<_> = data["gaussian_relative_log_density"]
        .as_array()
        .unwrap()
        .iter()
        // JSON prediction parameters cannot represent NaN or infinity. Those
        // independent distribution rows are not reachable prior inputs.
        .filter(|case| case["x"].is_number())
        .collect();
    assert_eq!(cases.len(), 15, "all finite-input prior references run");
    for case in cases {
        let payload = json!({"gaussian_priors": {
            "theta": {"mean": case["mean"], "sd": case["sd"]}
        }});
        let priors = GaussianPriors::parse(&payload, ["theta"].into_iter()).unwrap();
        assert_eq!(
            priors.label(),
            "independent_gaussian_or_uniform_within_bounds"
        );
        let prediction = json!({"parameters": {"theta": case["x"]}});
        assert_reference(priors.log_density(&prediction), case);
    }
}

#[test]
fn gaussian_prior_omission_and_invalid_domains_keep_their_contracts() {
    let payload = json!({});
    let prior = GaussianPriors::parse(&payload, ["theta"].into_iter()).unwrap();
    assert_eq!(prior.label(), "uniform_within_bounds");
    assert_eq!(prior.log_density(&json!({})), 0.);

    for invalid in [json!(0), json!(-1), json!(null), json!("1")] {
        let payload = json!({"gaussian_priors": {
            "theta": {"mean": 0., "sd": invalid}
        }});
        assert_eq!(
            GaussianPriors::parse(&payload, ["theta"].into_iter()).err(),
            Some("Gaussian prior sd must be finite and positive")
        );
    }
    for invalid in [json!(null), json!("0")] {
        let payload = json!({"gaussian_priors": {
            "theta": {"mean": invalid, "sd": 1.}
        }});
        assert_eq!(
            GaussianPriors::parse(&payload, ["theta"].into_iter()).err(),
            Some("Gaussian prior mean must be finite")
        );
    }
    let unknown = json!({"gaussian_priors": {"other": {"mean": 0., "sd": 1.}}});
    assert_eq!(
        GaussianPriors::parse(&unknown, ["theta"].into_iter()).err(),
        Some("Gaussian prior must name a sampled parameter")
    );
    let malformed = json!({"gaussian_priors": []});
    assert_eq!(
        GaussianPriors::parse(&malformed, ["theta"].into_iter()).err(),
        Some("gaussian_priors must be a parameter map")
    );
}

#[test]
fn likelihood_uses_actual_predictions_without_losing_small_or_scaled_scores() {
    let kernel = crate::KernelBinding::new();
    let data = fixture();
    let selected = [
        "near_center_1e10",
        "near_center_1e8",
        "standard_positive",
        "large_positive",
        "large_finite_log_density",
        "overflow_log_density",
        "shifted_scaled",
        "shifted_at_mean",
        "tiny_scale_unit_residual",
        "huge_scale_unit_residual",
    ];
    let mut checked = 0;
    for case in data["gaussian_relative_log_density"].as_array().unwrap() {
        if !selected.contains(&case["id"].as_str().unwrap()) {
            continue;
        }
        // At t=x0=0 the independent logistic identity is f(0)=L/2.
        // All selected x are positive, so these use valid positive capacities.
        let capacity = 2. * case["x"].as_f64().unwrap();
        let prediction = json!({"inputs": {"time": [0.]},
            "parameters": {"L": capacity, "k": 1., "x0": 0.}});
        let actual = log_likelihood(
            &kernel,
            "logistic",
            &prediction,
            &[case["mean"].as_f64().unwrap()],
            case["sd"].as_f64().unwrap(),
        )
        .unwrap();
        assert_reference(actual, case);
        checked += 1;
    }
    assert_eq!(checked, selected.len());

    let prediction = json!({"inputs": {"time": [0., 1.]},
        "parameters": {"L": 2., "k": 1., "x0": 0.}});
    // mpmath at 100 decimal digits: -[1 + (2/(1+exp(-1)))^2]/2.
    // The absolute tolerance allows binary64 predictor and summation rounding.
    let actual = log_likelihood(&kernel, "logistic", &prediction, &[0., 0.], 1.).unwrap();
    let expected = -1.568893290777046;
    assert!(
        (actual - expected).abs() <= 1e-14,
        "both observations contribute: actual={actual}, expected={expected}"
    );
    assert_eq!(
        log_likelihood(&kernel, "logistic", &prediction, &[0.], 1.),
        Err("posterior requires aligned scalar predictions")
    );
}

#[test]
fn likelihood_observation_validation_rejects_invalid_noise_and_data() {
    for invalid in [json!(0), json!(-1), json!(null), json!("1")] {
        let payload = json!({"noise_sd": invalid, "observed": [1.]});
        assert_eq!(
            observations(&payload).err(),
            Some("noise_sd must be finite and positive")
        );
    }
    for invalid in [json!([null]), json!(["1"]), json!(null)] {
        let payload = json!({"noise_sd": 1., "observed": invalid});
        assert_eq!(
            observations(&payload).err(),
            Some("observed must be a finite numeric array")
        );
    }
    for invalid in [json!([]), json!(vec![1.; 501])] {
        let payload = json!({"noise_sd": 1., "observed": invalid});
        assert_eq!(
            observations(&payload).err(),
            Some("posterior requires 1 to 500 finite observations")
        );
    }
}
