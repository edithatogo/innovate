use innovate_core::inference::posterior;
use innovate_core::models::hierarchical::{
    PanelFitRequest, fit_hierarchical, unseen_group_predictive,
};

#[test]
fn hierarchical_recovers_pooling_limits_and_shrinks_sparse_groups() {
    let outcome = fit_hierarchical(&PanelFitRequest {
        group_ids: vec![
            "dense".into(),
            "dense".into(),
            "dense".into(),
            "dense".into(),
            "sparse".into(),
        ],
        values: vec![10.0, 10.2, 9.8, 10.0, 2.0],
        prior_mean: 0.0,
        prior_sd: 100.0,
        obs_sd: 1.0,
        tau: 1.0,
    });
    assert!(posterior(&outcome).is_err());
    let row = match &outcome {
        innovate_core::inference::FitOutcome::Deterministic(res) => &res.values,
        other => panic!("expected deterministic fit, got {other:?}"),
    };
    let mu_hat = row[0];
    let theta_dense = row[2];
    let theta_sparse = row[3];

    // Dense group shrinks less than sparse group; both shrink strictly toward mu_hat.
    assert!(theta_dense > mu_hat && theta_dense < 10.0);
    assert!(theta_sparse > 2.0 && theta_sparse < mu_hat);
    assert!((10.0 - theta_dense) < (theta_sparse - 2.0));

    // Unseen group predictive uncertainty exceeds hypermean sd due to tau.
    let (pred_mean, pred_sd) = unseen_group_predictive(&outcome).unwrap();
    assert_eq!(pred_mean, mu_hat);
    assert!(pred_sd > row[1]);

    // Complete pooling limit (tau = 0) sets all groups equal to hypermean.
    let pooled = fit_hierarchical(&PanelFitRequest {
        group_ids: vec!["a".into(), "b".into()],
        values: vec![10.0, 2.0],
        prior_mean: 0.0,
        prior_sd: 100.0,
        obs_sd: 1.0,
        tau: 0.0,
    });
    let p_row = match &pooled {
        innovate_core::inference::FitOutcome::Deterministic(res) => &res.values,
        other => panic!("expected deterministic fit, got {other:?}"),
    };
    assert!((p_row[2] - p_row[0]).abs() < 1e-12);
    assert!((p_row[3] - p_row[0]).abs() < 1e-12);
}
