use innovate_rust::{json, KernelBinding, KernelRequest};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn logistic_fit_request(binding: &KernelBinding) -> KernelRequest {
    binding.fit_model_request(
        "logistic",
        json!({
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "observed": [
                    12.45533581839509,
                    21.416501696862853,
                    34.29895373044305,
                    50.0,
                    65.70104626955695,
                    78.58349830313714
                ]
            }
        }),
    )
}

fn logistic_predict_request(binding: &KernelBinding) -> KernelRequest {
    binding.predict_model_request(
        "logistic",
        json!({
            "state": {
                "model_key": "logistic",
                "model_name": "LogisticModel",
                "constructor_kwargs": {},
                "parameters": {
                    "L": 100.0,
                    "k": 0.65,
                    "x0": 3.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            }
        }),
    )
}

fn logistic_simulate_request(binding: &KernelBinding) -> KernelRequest {
    binding.simulate_model_request(
        "logistic",
        json!({
            "state": {
                "model_key": "logistic",
                "model_name": "LogisticModel",
                "constructor_kwargs": {},
                "parameters": {
                    "L": 120.0,
                    "k": 0.55,
                    "x0": 2.5
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            }
        }),
    )
}

fn logistic_summary_request(binding: &KernelBinding) -> KernelRequest {
    binding.summarize_model_request(
        "logistic",
        json!({
            "state": {
                "model_key": "logistic",
                "model_name": "LogisticModel",
                "constructor_kwargs": {},
                "parameters": {
                    "L": 100.0,
                    "k": 0.65,
                    "x0": 3.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "observed": [
                    12.45533581839509,
                    21.416501696862853,
                    34.29895373044305,
                    50.0,
                    65.70104626955695,
                    78.58349830313714
                ]
            }
        }),
    )
}

fn logistic_diagnose_request(binding: &KernelBinding) -> KernelRequest {
    binding.diagnose_model_request(
        "logistic",
        json!({
            "state": {
                "model_key": "logistic",
                "model_name": "LogisticModel",
                "constructor_kwargs": {},
                "parameters": {
                    "L": 100.0,
                    "k": 0.65,
                    "x0": 3.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "observed": [
                    12.45533581839509,
                    21.416501696862853,
                    34.29895373044305,
                    50.0,
                    65.70104626955695,
                    78.58349830313714
                ]
            }
        }),
    )
}

fn iteration_count() -> usize {
    std::env::var("INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10_000)
}

fn main() {
    let _profiler = dhat::Profiler::new_heap();
    let binding = KernelBinding::new();
    let iterations = iteration_count();

    let fit_request = logistic_fit_request(&binding);
    let predict_request = logistic_predict_request(&binding);
    let simulate_request = logistic_simulate_request(&binding);
    let summary_request = logistic_summary_request(&binding);
    let diagnose_request = logistic_diagnose_request(&binding);

    for _ in 0..iterations {
        let _ = binding
            .fit_model_native(&fit_request)
            .expect("native logistic fit should succeed");
        let _ = binding
            .predict_model_native(&predict_request)
            .expect("native logistic prediction should succeed");
        let _ = binding
            .simulate_model_native(&simulate_request)
            .expect("native logistic simulation should succeed");
        let _ = binding
            .summarize_model_native(&summary_request)
            .expect("native logistic summary should succeed");
        let _ = binding
            .diagnose_model_native(&diagnose_request)
            .expect("native logistic diagnostics should succeed");
    }
}
