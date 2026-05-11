use innovate_rust::{json, KernelBinding, KernelRequest};

const ITERATIONS_ENV: &str = "INNOVATE_RUST_MEMORY_PROFILE_ITERATIONS";
const OUTPUT_ENV: &str = "INNOVATE_RUST_MEMORY_PROFILE_OUTPUT";

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

fn bass_predict_request(binding: &KernelBinding) -> KernelRequest {
    binding.predict_model_request(
        "bass",
        json!({
            "state": {
                "model_key": "bass",
                "model_name": "BassModel",
                "constructor_kwargs": {},
                "parameters": {
                    "p": 0.03,
                    "q": 0.38,
                    "m": 120.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 0.8, 1.6, 2.4, 3.2, 4.0]
            }
        }),
    )
}

fn bass_fit_request(binding: &KernelBinding) -> KernelRequest {
    binding.fit_model_request(
        "bass",
        json!({
            "inputs": {
                "time": [0.0, 0.8, 1.6, 2.4, 3.2, 4.0],
                "observed": [0.02, 0.05, 0.11, 0.2, 0.33, 0.49]
            }
        }),
    )
}

fn bass_simulate_request(binding: &KernelBinding) -> KernelRequest {
    binding.simulate_model_request(
        "bass",
        json!({
            "state": {
                "model_key": "bass",
                "model_name": "BassModel",
                "constructor_kwargs": {},
                "parameters": {
                    "p": 0.02,
                    "q": 0.45,
                    "m": 150.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0]
            }
        }),
    )
}

fn bass_summary_request(binding: &KernelBinding) -> KernelRequest {
    binding.summarize_model_request(
        "bass",
        json!({
            "state": {
                "model_key": "bass",
                "model_name": "BassModel",
                "constructor_kwargs": {},
                "parameters": {
                    "p": 0.02,
                    "q": 0.45,
                    "m": 150.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.02, 0.05, 0.11, 0.2, 0.33]
            }
        }),
    )
}

fn bass_diagnose_request(binding: &KernelBinding) -> KernelRequest {
    binding.diagnose_model_request(
        "bass",
        json!({
            "state": {
                "model_key": "bass",
                "model_name": "BassModel",
                "constructor_kwargs": {},
                "parameters": {
                    "p": 0.02,
                    "q": 0.45,
                    "m": 150.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.02, 0.05, 0.11, 0.2, 0.33]
            }
        }),
    )
}

fn iteration_count() -> usize {
    std::env::var(ITERATIONS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(10_000)
}

fn output_file_name() -> String {
    std::env::var(OUTPUT_ENV).unwrap_or_else(|_| "dhat-native-kernels-heap.json".to_string())
}

fn main() {
    let _profiler = dhat::Profiler::builder()
        .file_name(output_file_name())
        .build();
    let binding = KernelBinding::new();
    let iterations = iteration_count();

    let fit_request = logistic_fit_request(&binding);
    let predict_request = logistic_predict_request(&binding);
    let simulate_request = logistic_simulate_request(&binding);
    let summary_request = logistic_summary_request(&binding);
    let diagnose_request = logistic_diagnose_request(&binding);
    let bass_fit_request = bass_fit_request(&binding);
    let bass_predict_request = bass_predict_request(&binding);
    let bass_simulate_request = bass_simulate_request(&binding);
    let bass_summary_request = bass_summary_request(&binding);
    let bass_diagnose_request = bass_diagnose_request(&binding);

    for _ in 0..iterations {
        let _ = binding
            .fit_model_native(&fit_request)
            .expect("native logistic fit should succeed");
        let _ = binding
            .fit_model_native(&bass_fit_request)
            .expect("native Bass fit should succeed");
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
        let _ = binding
            .summarize_model_native(&bass_summary_request)
            .expect("native Bass summary should succeed");
        let _ = binding
            .diagnose_model_native(&bass_diagnose_request)
            .expect("native Bass diagnostics should succeed");
        let _ = binding
            .predict_model_native(&bass_predict_request)
            .expect("native Bass prediction should succeed");
        let _ = binding
            .simulate_model_native(&bass_simulate_request)
            .expect("native Bass simulation should succeed");
    }
}
