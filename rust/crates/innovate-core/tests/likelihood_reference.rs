use innovate_core::domain::{DomainObservationSeries, ObservationKind, TimeGrid, Units};
use innovate_core::likelihood::{
    LikelihoodConfig, LikelihoodFamily, ObservationScale, log_likelihood,
};

#[test]
fn likelihoods_match_independent_values_and_handle_zero_and_large_counts() {
    let grid = TimeGrid::try_from(vec![0.0, 1.0, 2.0]).unwrap();
    let counts = DomainObservationSeries::try_new(
        grid.clone(),
        vec![Some(0.0), Some(3.0), Some(500.0)],
        ObservationKind::NewAdoption,
        Units::Count,
        None,
    )
    .unwrap();
    let means = vec![1.5, 2.5, 495.0];

    let pois = log_likelihood(
        &counts,
        &means,
        &LikelihoodConfig {
            family: LikelihoodFamily::Poisson,
            scale: ObservationScale::IncidenceCounts,
        },
    )
    .unwrap();
    // Exact check for Poisson: sum(y*ln(mu) - mu - ln(y!))
    let expected_pois = -1.5 + (3.0 * 2.5_f64.ln() - 2.5 - 6.0_f64.ln()) + {
        let mut lg = 0.0;
        for k in 1..=500 {
            lg += (k as f64).ln();
        }
        500.0 * 495.0_f64.ln() - 495.0 - lg
    };
    assert!((pois - expected_pois).abs() < 1e-8);

    // Negative binomial approaches Poisson as dispersion -> infinity.
    let nb_large_r = log_likelihood(
        &counts,
        &means,
        &LikelihoodConfig {
            family: LikelihoodFamily::NegativeBinomial { dispersion: 1e7 },
            scale: ObservationScale::IncidenceCounts,
        },
    )
    .unwrap();
    assert!((nb_large_r - pois).abs() < 1e-3);

    // Gaussian & Student-t continuous likelihoods
    let cont = DomainObservationSeries::try_new(
        grid,
        vec![Some(1.0), None, Some(3.0)],
        ObservationKind::Rate,
        Units::RatePerUnitTime,
        None,
    )
    .unwrap();
    let g_ll = log_likelihood(
        &cont,
        &[1.0, 2.0, 2.5],
        &LikelihoodConfig {
            family: LikelihoodFamily::Gaussian { sigma: 0.5 },
            scale: ObservationScale::Continuous,
        },
    )
    .unwrap();
    let t_ll = log_likelihood(
        &cont,
        &[1.0, 2.0, 2.5],
        &LikelihoodConfig {
            family: LikelihoodFamily::StudentT {
                sigma: 0.5,
                df: 4.0,
            },
            scale: ObservationScale::Continuous,
        },
    )
    .unwrap();
    assert!(g_ll.is_finite() && t_ll.is_finite());
}

#[test]
fn cumulative_counts_and_invalid_dispersion_are_rejected() {
    let grid = TimeGrid::try_from(vec![0.0, 1.0]).unwrap();
    let counts = DomainObservationSeries::try_new(
        grid,
        vec![Some(1.0), Some(3.0)],
        ObservationKind::ActiveStock,
        Units::Count,
        None,
    )
    .unwrap();
    assert!(
        log_likelihood(
            &counts,
            &[1.0, 2.0],
            &LikelihoodConfig {
                family: LikelihoodFamily::Poisson,
                scale: ObservationScale::CumulativeCounts,
            },
        )
        .is_err()
    );
    assert!(
        log_likelihood(
            &counts,
            &[1.0, 2.0],
            &LikelihoodConfig {
                family: LikelihoodFamily::NegativeBinomial { dispersion: 0.0 },
                scale: ObservationScale::IncidenceCounts,
            },
        )
        .is_err()
    );
}
