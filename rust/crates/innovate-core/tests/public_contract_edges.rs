use innovate_core::domain::{
    Denominator, DomainObservationSeries, ObservationKind, TimeGrid, Units,
};
use innovate_core::inference::FitOutcome;
use innovate_core::likelihood::{
    LikelihoodConfig, LikelihoodFamily, ObservationScale, log_likelihood,
};
use innovate_core::models::competition::{CompetitionParameters, MarketState, competing_rates};
use innovate_core::models::demand::{ChoiceSet, PreferenceParameters, choice_shares};
use innovate_core::models::expectation_feedback::{DelayedStateModel, ExternalLagSignal};
use innovate_core::models::network::neighbor_exposure;
use innovate_core::models::regimes::{DatedShock, RegimeRequest, predict_regimes};
use innovate_core::models::repeat_use::{
    ActiveUseState, SupplyIntensity, UseRates, expected_transactions, step_active_use,
};
use innovate_core::models::{
    ClassicalFamily, FitRequest, PredictionRequest, fit_classical, predict_classical,
};

fn observation(value: f64, kind: ObservationKind, unit: Units) -> DomainObservationSeries {
    DomainObservationSeries::try_new(
        TimeGrid::try_from(vec![0.0]).unwrap(),
        vec![Some(value)],
        kind,
        unit,
        None,
    )
    .unwrap()
}

#[test]
fn observation_contract_retains_metadata_and_rejects_incompatible_inputs() {
    let grid = TimeGrid::try_from(vec![1.0]).unwrap();
    let series = DomainObservationSeries::try_new(
        grid.clone(),
        vec![Some(0.25)],
        ObservationKind::Share,
        Units::Proportion,
        Some(Denominator::Population(100.0)),
    )
    .unwrap();
    assert!(!series.grid().is_empty());
    assert_eq!(series.grid().as_slice(), &[1.0]);
    assert_eq!(series.kind(), ObservationKind::Share);
    assert_eq!(series.units(), Units::Proportion);
    assert_eq!(series.denominator(), Some(Denominator::Population(100.0)));

    let bad_unit = DomainObservationSeries::try_new(
        grid.clone(),
        vec![Some(0.5)],
        ObservationKind::Share,
        Units::Count,
        None,
    );
    assert_eq!(bad_unit.unwrap_err().code, "incompatible_units");
    let nonfinite = DomainObservationSeries::try_new(
        grid,
        vec![Some(f64::NAN)],
        ObservationKind::Rate,
        Units::RatePerUnitTime,
        None,
    );
    assert_eq!(nonfinite.unwrap_err().code, "non_finite_observation");
}

#[test]
fn likelihoods_reject_invalid_counts_means_and_scales() {
    let count = observation(1.0, ObservationKind::NewAdoption, Units::Count);
    let fractional = observation(0.5, ObservationKind::NewAdoption, Units::Count);
    let config = |family| LikelihoodConfig {
        family,
        scale: ObservationScale::IncidenceCounts,
    };

    assert_eq!(
        log_likelihood(&count, &[], &config(LikelihoodFamily::Poisson))
            .unwrap_err()
            .code,
        "length_mismatch"
    );
    assert_eq!(
        log_likelihood(&count, &[f64::NAN], &config(LikelihoodFamily::Poisson))
            .unwrap_err()
            .code,
        "non_finite_input"
    );
    for family in [
        LikelihoodFamily::Poisson,
        LikelihoodFamily::NegativeBinomial { dispersion: 2.0 },
    ] {
        assert_eq!(
            log_likelihood(&fractional, &[1.0], &config(family))
                .unwrap_err()
                .code,
            "invalid_count"
        );
        assert_eq!(
            log_likelihood(&count, &[0.0], &config(family))
                .unwrap_err()
                .code,
            "invalid_mean"
        );
    }
    let zero = observation(0.0, ObservationKind::NewAdoption, Units::Count);
    let ll = log_likelihood(
        &zero,
        &[0.0],
        &config(LikelihoodFamily::NegativeBinomial { dispersion: 2.0 }),
    )
    .unwrap();
    assert_eq!(ll, 0.0);

    let continuous = observation(1.0, ObservationKind::Rate, Units::RatePerUnitTime);
    for family in [
        LikelihoodFamily::Gaussian { sigma: 0.0 },
        LikelihoodFamily::StudentT {
            sigma: 1.0,
            df: 0.0,
        },
    ] {
        assert_eq!(
            log_likelihood(&continuous, &[1.0], &config(family))
                .unwrap_err()
                .code,
            "invalid_scale"
        );
    }
    let huge = observation(1e308, ObservationKind::Rate, Units::RatePerUnitTime);
    assert_eq!(
        log_likelihood(
            &huge,
            &[0.0],
            &config(LikelihoodFamily::Gaussian { sigma: 1.0 })
        )
        .unwrap_err()
        .code,
        "numerical_overflow"
    );
}

#[test]
fn classical_predictions_and_regime_shocks_validate_and_match_closed_forms() {
    let grid = TimeGrid::try_from(vec![1.0]).unwrap();
    for family in [
        ClassicalFamily::Bass,
        ClassicalFamily::Logistic,
        ClassicalFamily::Gompertz,
        ClassicalFamily::FisherPry,
    ] {
        let request = PredictionRequest {
            family,
            time: grid.clone(),
            parameters: vec![],
        };
        assert_eq!(
            predict_classical(&request).unwrap_err().code,
            "invalid_dimension"
        );
        let fit = FitRequest {
            family,
            time: grid.clone(),
            observed: vec![],
        };
        assert!(matches!(fit_classical(&fit), FitOutcome::Invalid(_)));
    }

    let shock = DatedShock {
        effective_time: 0.0,
        p_multiplier: 2.0,
        q_multiplier: 1.0,
        capacity_multiplier: 0.5,
    };
    let cases = [
        (
            ClassicalFamily::Logistic,
            vec![100.0, 1.0, 0.0],
            50.0 / (1.0 + (-2.0_f64).exp()),
        ),
        (
            ClassicalFamily::Gompertz,
            vec![100.0, 1.0, 1.0],
            50.0 * (-(-2.0_f64).exp()).exp(),
        ),
        (
            ClassicalFamily::FisherPry,
            vec![1.0, 0.0],
            1.0 / (1.0 + (-2.0_f64).exp()),
        ),
    ];
    for (family, base_parameters, expected) in cases {
        let result = predict_regimes(&RegimeRequest {
            family,
            time: grid.clone(),
            base_parameters,
            shocks: vec![shock.clone()],
        })
        .unwrap();
        assert!((result.values[0] - expected).abs() < 1e-12);
    }
    let invalid = RegimeRequest {
        family: ClassicalFamily::Bass,
        time: grid,
        base_parameters: vec![0.03, 0.38, 1000.0],
        shocks: vec![DatedShock {
            p_multiplier: 0.0,
            ..shock
        }],
    };
    assert_eq!(predict_regimes(&invalid).unwrap_err().code, "invalid_shock");
}

#[test]
fn competition_choice_and_network_reject_invalid_shapes_and_values() {
    let state = MarketState {
        adoption: vec![100.0, 50.0],
    };
    let params = CompetitionParameters {
        market_capacity: 1000.0,
        entry_times: vec![0.0, 0.0],
        p: vec![0.02, 0.03],
        q: vec![0.3, 0.4],
        interaction: vec![vec![0.0, -0.1], vec![0.05, 0.0]],
    };
    let rates = competing_rates(1.0, &state, &params).unwrap();
    assert!((rates.rates[0] - 38.25).abs() < 1e-12);
    let mut malformed = params.clone();
    malformed.p.pop();
    assert_eq!(
        competing_rates(1.0, &state, &malformed).unwrap_err().code,
        "invalid_dimension"
    );
    malformed = params;
    malformed.market_capacity = 0.0;
    assert_eq!(
        competing_rates(1.0, &state, &malformed).unwrap_err().code,
        "invalid_parameter"
    );
    let choices = ChoiceSet {
        quality: vec![1.0],
        price: vec![],
    };
    assert_eq!(
        choice_shares(
            &choices,
            &PreferenceParameters {
                alpha_quality: 1.0,
                beta_price: -1.0
            }
        )
        .unwrap_err()
        .code,
        "invalid_choice_set"
    );
    assert_eq!(
        neighbor_exposure(&[vec![0.0]], &[1.0, 0.0])
            .unwrap_err()
            .code,
        "invalid_network"
    );
}

#[test]
fn lag_and_active_use_fail_closed_on_invalid_inputs() {
    let signal = ExternalLagSignal {
        lag: -1.0,
        baseline_value: 0.0,
    };
    assert_eq!(
        signal.evaluate(1.0, |t| t).unwrap_err().code,
        "invalid_lag_signal"
    );
    let signal = ExternalLagSignal {
        lag: 0.0,
        baseline_value: 0.0,
    };
    assert_eq!(
        signal.evaluate(1.0, |_| f64::NAN).unwrap_err().code,
        "non_finite_signal"
    );
    let delayed = DelayedStateModel {
        delay_steps: 0,
        feedback_gain: 1.0,
        decay: 0.1,
    };
    assert_eq!(
        delayed.step_hype(1.0, &[1.0], 0.0, -1.0).unwrap_err().code,
        "invalid_delayed_state"
    );
    assert_eq!(
        delayed
            .step_hype(1.0, &[f64::NAN], 0.0, 1.0)
            .unwrap_err()
            .code,
        "non_finite_history"
    );

    let state = ActiveUseState {
        active: vec![1.0],
        cumulative_starts: vec![1.0],
        cumulative_stops: vec![0.0],
    };
    let rates = UseRates {
        starts: vec![0.0],
        stops: vec![2.0],
        switch: vec![vec![0.0]],
    };
    assert_eq!(
        step_active_use(&state, &rates, 1.0).unwrap_err().code,
        "negative_stock"
    );
    assert_eq!(
        step_active_use(&state, &rates, -1.0).unwrap_err().code,
        "invalid_input"
    );
    let wrong_shape = UseRates {
        starts: vec![],
        ..rates
    };
    assert_eq!(
        step_active_use(&state, &wrong_shape, 1.0).unwrap_err().code,
        "invalid_dimension"
    );
    assert_eq!(
        expected_transactions(
            &state,
            &SupplyIntensity {
                prescriptions_per_active: vec![]
            }
        )
        .unwrap_err()
        .code,
        "invalid_dimension"
    );
    assert_eq!(
        expected_transactions(
            &state,
            &SupplyIntensity {
                prescriptions_per_active: vec![-1.0]
            }
        )
        .unwrap_err()
        .code,
        "invalid_supply_intensity"
    );
}
