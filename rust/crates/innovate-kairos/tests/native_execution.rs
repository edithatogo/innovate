use innovate_core::domain::TimeGrid;
use innovate_kairos::{KairosRuntime, SimulationRequest, run_innovation};

#[test]
fn native_kairos_execution_replays_deterministically_and_conserves_population() {
    let req = SimulationRequest {
        model_key: "bass".into(),
        time: TimeGrid::try_from(vec![0.0, 1.0, 5.0, 20.0]).unwrap(),
        p: 0.2,
        q: 0.3,
        m: 100.0,
    };
    let runtime = KairosRuntime {
        engine: "kairos_des".into(),
        seed: 42,
        population: 100,
        ticks_per_unit: 1_000_000,
    };

    let a = run_innovation(&req, &runtime).unwrap();
    let b = run_innovation(&req, &runtime).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.engine, "kairos_des");
    assert_eq!(a.counts.len(), 4);
    assert!(a.counts.windows(2).all(|w| w[0] <= w[1]));
    assert!(a.counts.iter().all(|&c| c <= runtime.population));

    let bad_runtime = KairosRuntime {
        engine: "python_fallback".into(),
        ..runtime
    };
    assert!(run_innovation(&req, &bad_runtime).is_err());
}
