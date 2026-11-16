//! Accuracy-controlled sequential performance workload runner (V04).

use innovate_core::domain::{TimeGrid, bass_cumulative};
use innovate_core::models::{ClassicalFamily, PredictionRequest, predict_classical};
use std::time::Instant;

fn main() {
    let time = TimeGrid::try_from((0..100).map(|i| i as f64 * 0.25).collect::<Vec<_>>()).unwrap();
    let t0 = Instant::now();
    let bundle = predict_classical(&PredictionRequest {
        family: ClassicalFamily::Bass,
        time: time.clone(),
        parameters: vec![0.03, 0.38, 1000.0],
    })
    .unwrap();
    let elapsed_ns = t0.elapsed().as_nanos();

    let mut max_err = 0.0_f64;
    for (&t, &v) in time.as_slice().iter().zip(&bundle.values) {
        let ref_val = bass_cumulative(t, 0.03, 0.38, 1000.0).unwrap();
        max_err = max_err.max((v - ref_val).abs());
    }
    assert!(max_err < 1e-10, "accuracy gate failed: max_err={max_err}");
    println!(
        "{{\"workload\":\"bass_predict_100\",\"elapsed_ns\":{elapsed_ns},\"max_abs_error\":{max_err}}}"
    );
}
