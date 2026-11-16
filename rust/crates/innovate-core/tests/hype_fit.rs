use innovate_core::domain::TimeGrid;
use innovate_core::inference::hype_fit::{HypeFitRequest, fit_hype};
use innovate_core::inference::posterior;
use innovate_core::models::hype::{HypeConfiguration, HypeParameters, HypePulse, hype_mean};

#[test]
fn hype_fit_recovers_noiseless_curve_and_flags_overlapping_pulses() {
    let true_p = HypeParameters {
        baseline: 0.5,
        maturity_amplitude: 2.0,
        growth: 0.8,
        t_maturity: 7.0,
        pulses: vec![HypePulse {
            amplitude: 3.0,
            center: 3.0,
            width: 1.0,
        }],
        configuration: HypeConfiguration::OrderedCycle,
    };
    let time = TimeGrid::try_from((0..16).map(|i| i as f64 * 0.7).collect::<Vec<_>>()).unwrap();
    let observed: Vec<f64> = time
        .as_slice()
        .iter()
        .map(|&t| hype_mean(t, &true_p).unwrap())
        .collect();

    let guess = HypeParameters {
        baseline: 0.2,
        maturity_amplitude: 1.6,
        growth: 0.6,
        t_maturity: 6.5,
        pulses: vec![HypePulse {
            amplitude: 2.4,
            center: 2.8,
            width: 1.2,
        }],
        configuration: HypeConfiguration::OrderedCycle,
    };
    let outcome = fit_hype(&HypeFitRequest {
        time: time.clone(),
        observed: observed.clone(),
        initial_guesses: vec![guess],
        max_iterations: 200,
    });
    assert_eq!(posterior(&outcome).unwrap_err().code, "deterministic_fit");
    let row = match &outcome {
        innovate_core::inference::FitOutcome::Deterministic(res) => &res.values,
        other => panic!("expected deterministic fit, got {other:?}"),
    };
    let fitted = HypeParameters {
        baseline: row[0],
        maturity_amplitude: row[1],
        growth: row[2],
        t_maturity: row[3],
        pulses: vec![HypePulse {
            amplitude: row[4],
            center: row[5],
            width: row[6],
        }],
        configuration: HypeConfiguration::OrderedCycle,
    };
    for (&t, &y) in time.as_slice().iter().zip(&observed) {
        assert!((hype_mean(t, &fitted).unwrap() - y).abs() < 1e-3);
    }

    // Overlapping pulses must be flagged in diagnostics.
    let overlap_guess = HypeParameters {
        baseline: 0.5,
        maturity_amplitude: 2.0,
        growth: 0.8,
        t_maturity: 7.0,
        pulses: vec![
            HypePulse {
                amplitude: 1.5,
                center: 3.0,
                width: 1.5,
            },
            HypePulse {
                amplitude: 1.5,
                center: 3.1,
                width: 1.5,
            },
        ],
        configuration: HypeConfiguration::OrderedCycle,
    };
    let overlap_out = fit_hype(&HypeFitRequest {
        time,
        observed,
        initial_guesses: vec![overlap_guess],
        max_iterations: 10,
    });
    let innovate_core::inference::FitOutcome::Deterministic(res) = overlap_out else {
        panic!("expected a deterministic overlap fit");
    };
    assert!(res.diagnostics.contains("overlapping_pulses=true"));
}
