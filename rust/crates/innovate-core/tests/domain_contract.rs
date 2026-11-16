use innovate_core::domain::{
    Denominator, DomainObservationSeries, ObservationKind, TimeGrid, Units,
};

#[test]
fn time_grid_rejects_empty_non_finite_unsorted_and_duplicate_times() {
    assert!(TimeGrid::try_from(vec![]).is_err());
    assert!(TimeGrid::try_from(vec![0.0, f64::NAN]).is_err());
    assert!(TimeGrid::try_from(vec![0.0, f64::INFINITY]).is_err());
    assert!(TimeGrid::try_from(vec![2.0, 1.0]).is_err());
    assert!(TimeGrid::try_from(vec![1.0, 1.0]).is_err());
    assert!(TimeGrid::try_from(vec![0.0, 1.0, 2.5]).is_ok());
}

#[test]
fn domain_observation_series_enforces_contracts_and_preserves_missingness() {
    let grid = TimeGrid::try_from(vec![0.0, 1.0, 2.0]).unwrap();
    let series = DomainObservationSeries::try_new(
        grid.clone(),
        vec![None, Some(0.0), Some(0.4)],
        ObservationKind::Share,
        Units::Proportion,
        Some(Denominator::Population(10_000.0)),
    )
    .expect("valid share series");
    assert_eq!(series.values()[0], None);
    assert_eq!(series.values()[1], Some(0.0));

    // Length mismatch, negative count, share > 1, incompatible denominator, non-positive denominator
    assert!(
        DomainObservationSeries::try_new(
            grid.clone(),
            vec![Some(1.0)],
            ObservationKind::NewAdoption,
            Units::Count,
            None,
        )
        .is_err()
    );
    assert!(
        DomainObservationSeries::try_new(
            grid.clone(),
            vec![Some(-1.0), Some(0.0), Some(1.0)],
            ObservationKind::ActiveStock,
            Units::Count,
            None,
        )
        .is_err()
    );
    assert!(
        DomainObservationSeries::try_new(
            grid.clone(),
            vec![Some(0.2), Some(1.2), Some(0.5)],
            ObservationKind::Share,
            Units::Proportion,
            None,
        )
        .is_err()
    );
    assert!(
        DomainObservationSeries::try_new(
            grid.clone(),
            vec![Some(1.0), Some(2.0), Some(3.0)],
            ObservationKind::Transactions,
            Units::Count,
            Some(Denominator::MarketPotential(100.0)),
        )
        .is_err()
    );
    assert!(
        DomainObservationSeries::try_new(
            grid,
            vec![Some(0.1), Some(0.2), Some(0.3)],
            ObservationKind::Share,
            Units::Proportion,
            Some(Denominator::Population(0.0)),
        )
        .is_err()
    );
}
