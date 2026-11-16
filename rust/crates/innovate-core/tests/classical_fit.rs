use innovate_core::domain::TimeGrid;
use innovate_core::inference::posterior;
use innovate_core::models::{
    ClassicalFamily, FitRequest, PredictionRequest, fit_classical, predict_classical,
};

#[test]
fn classical_predict_and_fit_recover_noiseless_families() {
    let time = TimeGrid::try_from((0..15).map(|i| i as f64).collect::<Vec<_>>()).unwrap();

    // Bass prediction & fit recovery
    let bass_pred = predict_classical(&PredictionRequest {
        family: ClassicalFamily::Bass,
        time: time.clone(),
        parameters: vec![0.03, 0.38, 1000.0],
    })
    .unwrap();
    assert_eq!(bass_pred.values[0], 0.0);
    assert!(bass_pred.values.windows(2).all(|w| w[1] >= w[0]));
    let bass_fit = fit_classical(&FitRequest {
        family: ClassicalFamily::Bass,
        time: time.clone(),
        observed: bass_pred.values.clone(),
    });
    assert!(matches!(
        bass_fit,
        innovate_core::inference::FitOutcome::Deterministic(_)
    ));
    assert_eq!(posterior(&bass_fit).unwrap_err().code, "deterministic_fit");

    // Logistic, Gompertz, Fisher-Pry predictions
    let log_pred = predict_classical(&PredictionRequest {
        family: ClassicalFamily::Logistic,
        time: time.clone(),
        parameters: vec![100.0, 0.5, 6.0],
    })
    .unwrap();
    assert!((log_pred.values[6] - 50.0).abs() < 1e-12);

    let gomp_pred = predict_classical(&PredictionRequest {
        family: ClassicalFamily::Gompertz,
        time: time.clone(),
        parameters: vec![200.0, 2.0, 0.3],
    })
    .unwrap();
    assert!(gomp_pred.values.iter().all(|&v| (0.0..=200.0).contains(&v)));

    let fp_pred = predict_classical(&PredictionRequest {
        family: ClassicalFamily::FisherPry,
        time: time.clone(),
        parameters: vec![0.6, 5.0],
    })
    .unwrap();
    assert!((fp_pred.values[5] - 0.5).abs() < 1e-12);
}

#[test]
fn classical_flags_short_pre_inflection_and_rejects_invalid_inputs() {
    let short_time = TimeGrid::try_from(vec![0.0, 1.0, 2.0]).unwrap();
    let short_fit = fit_classical(&FitRequest {
        family: ClassicalFamily::Bass,
        time: short_time.clone(),
        observed: vec![0.0, 5.0, 12.0],
    });
    if let innovate_core::inference::FitOutcome::Deterministic(res) = short_fit {
        assert!(res.diagnostics.contains("weakly_identified"));
    }

    assert!(
        predict_classical(&PredictionRequest {
            family: ClassicalFamily::Bass,
            time: short_time,
            parameters: vec![0.03, 0.38],
        })
        .is_err()
    );
}
