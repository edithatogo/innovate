use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use innovate_rust::{json, KernelBinding, KernelRequest};

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

fn bench_native_logistic_paths(c: &mut Criterion) {
    let binding = KernelBinding::new();
    let fit_request = logistic_fit_request(&binding);
    let predict_request = logistic_predict_request(&binding);
    let simulate_request = logistic_simulate_request(&binding);
    let summary_request = logistic_summary_request(&binding);
    let diagnose_request = logistic_diagnose_request(&binding);

    let mut group = c.benchmark_group("native_logistic_kernel");
    group.sample_size(20);

    group.bench_function(BenchmarkId::new("fit_model_native", "logistic"), |b| {
        b.iter(|| {
            let response = binding
                .fit_model_native(black_box(&fit_request))
                .expect("native logistic fit should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("predict_model_native", "logistic"), |b| {
        b.iter(|| {
            let response = binding
                .predict_model_native(black_box(&predict_request))
                .expect("native logistic prediction should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("simulate_model_native", "logistic"), |b| {
        b.iter(|| {
            let response = binding
                .simulate_model_native(black_box(&simulate_request))
                .expect("native logistic simulation should succeed");
            black_box(response);
        })
    });

    group.bench_function(
        BenchmarkId::new("summarize_model_native", "logistic"),
        |b| {
            b.iter(|| {
                let response = binding
                    .summarize_model_native(black_box(&summary_request))
                    .expect("native logistic summary should succeed");
                black_box(response);
            })
        },
    );

    group.bench_function(BenchmarkId::new("diagnose_model_native", "logistic"), |b| {
        b.iter(|| {
            let response = binding
                .diagnose_model_native(black_box(&diagnose_request))
                .expect("native logistic diagnostics should succeed");
            black_box(response);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_native_logistic_paths);
criterion_main!(benches);
