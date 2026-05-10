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

fn gompertz_fit_request(binding: &KernelBinding) -> KernelRequest {
    binding.fit_model_request(
        "gompertz",
        json!({
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.08, 0.15, 0.25, 0.4, 0.55]
            }
        }),
    )
}

fn gompertz_predict_request(binding: &KernelBinding) -> KernelRequest {
    binding.predict_model_request(
        "gompertz",
        json!({
            "state": {
                "model_key": "gompertz",
                "model_name": "GompertzModel",
                "constructor_kwargs": {},
                "parameters": {
                    "a": 100.0,
                    "b": 3.0,
                    "c": 0.3
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0]
            }
        }),
    )
}

fn gompertz_simulate_request(binding: &KernelBinding) -> KernelRequest {
    binding.simulate_model_request(
        "gompertz",
        json!({
            "state": {
                "model_key": "gompertz",
                "model_name": "GompertzModel",
                "constructor_kwargs": {},
                "parameters": {
                    "a": 100.0,
                    "b": 3.0,
                    "c": 0.3
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0]
            }
        }),
    )
}

fn gompertz_summary_request(binding: &KernelBinding) -> KernelRequest {
    binding.summarize_model_request(
        "gompertz",
        json!({
            "state": {
                "model_key": "gompertz",
                "model_name": "GompertzModel",
                "constructor_kwargs": {},
                "parameters": {
                    "a": 100.0,
                    "b": 3.0,
                    "c": 0.3
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.08, 0.15, 0.25, 0.4, 0.55]
            }
        }),
    )
}

fn gompertz_diagnose_request(binding: &KernelBinding) -> KernelRequest {
    binding.diagnose_model_request(
        "gompertz",
        json!({
            "state": {
                "model_key": "gompertz",
                "model_name": "GompertzModel",
                "constructor_kwargs": {},
                "parameters": {
                    "a": 100.0,
                    "b": 3.0,
                    "c": 0.3
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.08, 0.15, 0.25, 0.4, 0.55]
            }
        }),
    )
}

fn fisher_pry_fit_request(binding: &KernelBinding) -> KernelRequest {
    binding.fit_model_request(
        "fisher_pry",
        json!({
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.08, 0.15, 0.25, 0.4, 0.55]
            }
        }),
    )
}

fn fisher_pry_predict_request(binding: &KernelBinding) -> KernelRequest {
    binding.predict_model_request(
        "fisher_pry",
        json!({
            "state": {
                "model_key": "fisher_pry",
                "model_name": "FisherPryModel",
                "constructor_kwargs": {},
                "parameters": {
                    "alpha": 1.6,
                    "t0": 2.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0]
            }
        }),
    )
}

fn fisher_pry_simulate_request(binding: &KernelBinding) -> KernelRequest {
    binding.simulate_model_request(
        "fisher_pry",
        json!({
            "state": {
                "model_key": "fisher_pry",
                "model_name": "FisherPryModel",
                "constructor_kwargs": {},
                "parameters": {
                    "alpha": 1.6,
                    "t0": 2.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0]
            }
        }),
    )
}

fn fisher_pry_summary_request(binding: &KernelBinding) -> KernelRequest {
    binding.summarize_model_request(
        "fisher_pry",
        json!({
            "state": {
                "model_key": "fisher_pry",
                "model_name": "FisherPryModel",
                "constructor_kwargs": {},
                "parameters": {
                    "alpha": 1.6,
                    "t0": 2.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.08, 0.15, 0.25, 0.4, 0.55]
            }
        }),
    )
}

fn fisher_pry_diagnose_request(binding: &KernelBinding) -> KernelRequest {
    binding.diagnose_model_request(
        "fisher_pry",
        json!({
            "state": {
                "model_key": "fisher_pry",
                "model_name": "FisherPryModel",
                "constructor_kwargs": {},
                "parameters": {
                    "alpha": 1.6,
                    "t0": 2.0
                },
                "predict_kwargs": {}
            },
            "inputs": {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "observed": [0.08, 0.15, 0.25, 0.4, 0.55]
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

fn bench_native_logistic_paths(c: &mut Criterion) {
    let binding = KernelBinding::new();
    let fit_request = logistic_fit_request(&binding);
    let predict_request = logistic_predict_request(&binding);
    let simulate_request = logistic_simulate_request(&binding);
    let summary_request = logistic_summary_request(&binding);
    let diagnose_request = logistic_diagnose_request(&binding);
    let gompertz_fit_request = gompertz_fit_request(&binding);
    let gompertz_predict_request = gompertz_predict_request(&binding);
    let gompertz_simulate_request = gompertz_simulate_request(&binding);
    let gompertz_summary_request = gompertz_summary_request(&binding);
    let gompertz_diagnose_request = gompertz_diagnose_request(&binding);
    let fisher_pry_fit_request = fisher_pry_fit_request(&binding);
    let fisher_pry_predict_request = fisher_pry_predict_request(&binding);
    let fisher_pry_simulate_request = fisher_pry_simulate_request(&binding);
    let fisher_pry_summary_request = fisher_pry_summary_request(&binding);
    let fisher_pry_diagnose_request = fisher_pry_diagnose_request(&binding);
    let bass_predict_request = bass_predict_request(&binding);
    let bass_simulate_request = bass_simulate_request(&binding);

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

    group.bench_function(BenchmarkId::new("fit_model_native", "gompertz"), |b| {
        b.iter(|| {
            let response = binding
                .fit_model_native(black_box(&gompertz_fit_request))
                .expect("native Gompertz fit should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("fit_model_native", "fisher_pry"), |b| {
        b.iter(|| {
            let response = binding
                .fit_model_native(black_box(&fisher_pry_fit_request))
                .expect("native Fisher-Pry fit should succeed");
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

    group.bench_function(BenchmarkId::new("predict_model_native", "gompertz"), |b| {
        b.iter(|| {
            let response = binding
                .predict_model_native(black_box(&gompertz_predict_request))
                .expect("native Gompertz prediction should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("predict_model_native", "fisher_pry"), |b| {
        b.iter(|| {
            let response = binding
                .predict_model_native(black_box(&fisher_pry_predict_request))
                .expect("native Fisher-Pry prediction should succeed");
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

    group.bench_function(BenchmarkId::new("simulate_model_native", "gompertz"), |b| {
        b.iter(|| {
            let response = binding
                .simulate_model_native(black_box(&gompertz_simulate_request))
                .expect("native Gompertz simulation should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("simulate_model_native", "fisher_pry"), |b| {
        b.iter(|| {
            let response = binding
                .simulate_model_native(black_box(&fisher_pry_simulate_request))
                .expect("native Fisher-Pry simulation should succeed");
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

    group.bench_function(BenchmarkId::new("summarize_model_native", "gompertz"), |b| {
        b.iter(|| {
            let response = binding
                .summarize_model_native(black_box(&gompertz_summary_request))
                .expect("native Gompertz summary should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("summarize_model_native", "fisher_pry"), |b| {
        b.iter(|| {
            let response = binding
                .summarize_model_native(black_box(&fisher_pry_summary_request))
                .expect("native Fisher-Pry summary should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("diagnose_model_native", "logistic"), |b| {
        b.iter(|| {
            let response = binding
                .diagnose_model_native(black_box(&diagnose_request))
                .expect("native logistic diagnostics should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("diagnose_model_native", "gompertz"), |b| {
        b.iter(|| {
            let response = binding
                .diagnose_model_native(black_box(&gompertz_diagnose_request))
                .expect("native Gompertz diagnostics should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("diagnose_model_native", "fisher_pry"), |b| {
        b.iter(|| {
            let response = binding
                .diagnose_model_native(black_box(&fisher_pry_diagnose_request))
                .expect("native Fisher-Pry diagnostics should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("predict_model_native", "bass"), |b| {
        b.iter(|| {
            let response = binding
                .predict_model_native(black_box(&bass_predict_request))
                .expect("native Bass prediction should succeed");
            black_box(response);
        })
    });

    group.bench_function(BenchmarkId::new("simulate_model_native", "bass"), |b| {
        b.iter(|| {
            let response = binding
                .simulate_model_native(black_box(&bass_simulate_request))
                .expect("native Bass simulation should succeed");
            black_box(response);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_native_logistic_paths);
criterion_main!(benches);
