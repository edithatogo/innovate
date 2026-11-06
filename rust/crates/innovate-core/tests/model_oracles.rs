use innovate_core::{KernelBinding, KernelRequest, json};
use serde_json::Value;

fn numbers(result: &Value) -> Vec<f64> {
    if let Some(v) = result["values"].as_array() {
        v.iter().map(|v| v.as_f64().unwrap()).collect()
    } else {
        result["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v[0].as_f64().unwrap())
            .collect()
    }
}

#[test]
fn five_model_prediction_simulation_and_fitted_forecast_oracles() {
    let time: Vec<f64> = (0..21).map(f64::from).collect();
    let bass: Vec<f64> = time
        .iter()
        .map(|t| {
            let decay = (-0.38 * t).exp();
            100.0 * (1.0 - decay) / (1.0 + 0.35 / 0.03 * decay)
        })
        .collect();
    let models = [
        (
            "logistic",
            json!({"L":100.0,"k":0.6,"x0":8.0}),
            time.iter()
                .map(|t| 100.0 / (1.0 + (-0.6 * (t - 8.0)).exp()))
                .collect::<Vec<_>>(),
        ),
        (
            "gompertz",
            json!({"a":100.0,"b":3.0,"c":0.3}),
            time.iter()
                .map(|t| 100.0 * (-3.0 * (-0.3 * t).exp()).exp())
                .collect::<Vec<_>>(),
        ),
        (
            "fisher_pry",
            json!({"alpha":0.6,"t0":8.0}),
            time.iter()
                .map(|t| 1.0 / (1.0 + (-0.6 * (t - 8.0)).exp()))
                .collect::<Vec<_>>(),
        ),
        ("bass", json!({"p":0.03,"q":0.35,"m":100.0}), bass.clone()),
        ("norton_bass", json!({"p1":0.03,"q1":0.35,"m1":100.0}), bass),
    ];
    let binding = KernelBinding::new();
    for (model, parameters, observed) in models {
        let state = json!({"model_key":model,"parameters":parameters,"constructor_kwargs":{},"predict_kwargs":{}});
        let payload = json!({"inputs":{"time":time,"observed":observed},"state":state});
        for request in [
            KernelRequest::predict_model(model, payload.clone()),
            KernelRequest::simulate_model(model, payload.clone()),
        ] {
            let r = binding.invoke(&request).unwrap();
            assert_eq!(r.metadata["runtime"], "rust_native");
            for (got, want) in numbers(&r.result.unwrap()).iter().zip(&observed) {
                assert!((got - want).abs() < 1e-9, "{model}: {got} != {want}");
            }
        }
        let fit = binding
            .fit_model(model, json!({"inputs":{"time":time,"observed":observed}}))
            .unwrap()
            .result
            .unwrap();
        let predicted = binding
            .predict_model(model, json!({"inputs":{"time":time},"state":fit["state"]}))
            .unwrap()
            .result
            .unwrap();
        for (got, want) in numbers(&predicted).iter().zip(&observed) {
            assert!(
                (got - want).abs() < 1.0,
                "{model} fitted forecast: {got} != {want}"
            );
        }
        for request in [
            KernelRequest::summarize_model(model, payload.clone()),
            KernelRequest::diagnose_model(model, payload.clone()),
        ] {
            let result = binding.invoke(&request).unwrap();
            assert_eq!(result.metadata["runtime"], "rust_native");
            assert!(result.error.is_none());
        }
    }
}
