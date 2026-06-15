use innovate_rust::{json, KernelBinding, KernelOperation, KernelResponse};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use tracing_subscriber::fmt::MakeWriter;

#[derive(Clone)]
struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("buffer mutex should not be poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
struct SharedBufferWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> MakeWriter<'a> for SharedBufferWriter {
    type Writer = SharedBuffer;

    fn make_writer(&'a self) -> Self::Writer {
        SharedBuffer(self.0.clone())
    }
}

fn assert_json_numeric_map_has_finite_values(
    native: &serde_json::Value,
    bridged: &serde_json::Value,
) {
    let native_map = native
        .as_object()
        .expect("native diagnostics metrics should be an object");
    let bridged_map = bridged
        .as_object()
        .expect("bridged diagnostics metrics should be an object");
    assert_eq!(native_map.len(), bridged_map.len());

    for (key, native_value) in native_map {
        let bridged_value = bridged_map
            .get(key)
            .unwrap_or_else(|| panic!("bridged diagnostics metrics should contain key '{key}'"));
        let native_value = native_value
            .as_f64()
            .unwrap_or_else(|| panic!("native diagnostics metric '{key}' should be numeric"));
        let bridged_value = bridged_value
            .as_f64()
            .unwrap_or_else(|| panic!("bridged diagnostics metric '{key}' should be numeric"));
        assert!(
            native_value.is_finite(),
            "native diagnostics metric '{key}' should be finite"
        );
        assert!(
            bridged_value.is_finite(),
            "bridged diagnostics metric '{key}' should be finite"
        );
    }
}

fn bass_analysis_payload() -> serde_json::Value {
    json!({
        "state": {
            "model_key": "bass",
            "model_name": "BassModel",
            "constructor_kwargs": {},
            "parameters": {
                "p": 0.025,
                "q": 0.41,
                "m": 140.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 0.75, 1.5, 2.25, 3.0, 3.75],
            "observed": [0.0, 7.8, 20.6, 39.5, 61.4, 84.9]
        }
    })
}

fn bass_fit_payload() -> serde_json::Value {
    json!({
        "inputs": {
            "time": [0.0, 0.75, 1.5, 2.25, 3.0, 3.75],
            "observed": [0.0, 7.8, 20.6, 39.5, 61.4, 84.9]
        }
    })
}

fn norton_bass_prediction_payload(n_generations: u64, time: Vec<f64>) -> serde_json::Value {
    json!({
        "state": {
            "model_key": "norton_bass",
            "model_name": "NortonBassModel",
            "constructor_kwargs": {
                "n_generations": n_generations,
                "covariates": []
            },
            "parameters": {
                "p1": 0.001,
                "q1": 0.1,
                "m1": 100.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": time
        }
    })
}

fn norton_bass_analysis_payload(n_generations: u64) -> serde_json::Value {
    json!({
        "state": {
            "model_key": "norton_bass",
            "model_name": "NortonBassModel",
            "constructor_kwargs": {
                "n_generations": n_generations,
                "covariates": []
            },
            "parameters": {
                "p1": 0.001,
                "q1": 0.1,
                "m1": 100.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    })
}

#[test]
fn diagnostics_summary_extracts_support_information() {
    let response = KernelResponse {
        schema_version: "1.0".to_string(),
        operation: KernelOperation::DiagnoseModel,
        model_key: Some("logistic".to_string()),
        result: Some(json!({
            "diagnostics": {
                "support_level": "supported",
                "provenance": "deterministic",
                "comparison_family": "fitted",
                "warnings": [
                    {
                        "code": "nonlinear_fit_warning"
                    }
                ],
                "metrics": {
                    "RMSE": 0.12,
                    "MAE": 0.08
                },
                "model_name": "LogisticModel"
            },
            "state": {
                "model_key": "logistic"
            }
        })),
        error: None,
        metadata: json!({}),
    };

    let summary = response
        .diagnostics_summary()
        .expect("diagnostics summary should be present");
    assert_eq!(summary.support_level, "supported");
    assert_eq!(summary.provenance, "deterministic");
    assert_eq!(summary.comparison_family, "fitted");
    assert_eq!(summary.warning_count, 1);
    assert_eq!(summary.metric_count, 2);
    assert_eq!(summary.model_name.as_deref(), Some("LogisticModel"));
}

#[test]
fn request_builders_preserve_model_identity_and_payload_shape() {
    let binding = KernelBinding::new();
    let payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    });

    let predict_request = binding.predict_model_request("logistic", payload.clone());
    let simulate_request = binding.simulate_model_request("logistic", payload.clone());
    let summarize_request = binding.summarize_model_request("logistic", payload.clone());
    let diagnose_request = binding.diagnose_model_request("logistic", payload);

    assert_eq!(predict_request.operation, KernelOperation::PredictModel);
    assert_eq!(simulate_request.operation, KernelOperation::SimulateModel);
    assert_eq!(summarize_request.operation, KernelOperation::SummarizeModel);
    assert_eq!(diagnose_request.operation, KernelOperation::DiagnoseModel);

    assert_eq!(predict_request.model_key.as_deref(), Some("logistic"));
    assert_eq!(simulate_request.model_key.as_deref(), Some("logistic"));
    assert_eq!(summarize_request.model_key.as_deref(), Some("logistic"));
    assert_eq!(diagnose_request.model_key.as_deref(), Some("logistic"));
}

#[test]
fn native_logistic_fit_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let true_params = json!({
        "L": 100.0,
        "k": 0.65,
        "x0": 3.0
    });
    let time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let observed = vec![
        12.45533581839509,
        21.416501696862853,
        34.29895373044305,
        50.0,
        65.70104626955695,
        78.58349830313714,
    ];
    let payload = json!({
        "inputs": {
            "time": time,
            "observed": observed,
        }
    });

    let request = binding.fit_model_request("logistic", payload.clone());
    let native = binding
        .fit_model_native(&request)
        .expect("native logistic fit should succeed");
    let bridged = binding
        .fit_model_via_bridge("logistic", payload)
        .expect("Python bridge fit should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::FitModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_summary = native
        .diagnostics_summary()
        .expect("native logistic fit should expose diagnostics summary");
    let bridged_summary = bridged
        .diagnostics_summary()
        .expect("bridged logistic fit should expose diagnostics summary");
    assert_eq!(native_summary, bridged_summary);
    assert_eq!(native_summary.support_level, "supported");
    assert_eq!(native_summary.provenance, "deterministic");
    assert_eq!(native_summary.comparison_family, "fitted");
    assert_eq!(native_summary.model_name.as_deref(), Some("LogisticModel"));

    let native_result = native.result.expect("native fit should include a result");
    let bridged_result = bridged.result.expect("bridged fit should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);

    for key in ["L", "k", "x0"] {
        let native_value = native_result["parameters"][key]
            .as_f64()
            .expect("native parameter should be numeric");
        let bridged_value = bridged_result["parameters"][key]
            .as_f64()
            .expect("bridged parameter should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-3,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }

    let native_predictions = native_result["predictions"]["values"]
        .as_array()
        .expect("native predictions should be an array");
    let bridged_predictions = bridged_result["predictions"]["values"]
        .as_array()
        .expect("bridged predictions should be an array");
    assert_eq!(native_predictions.len(), bridged_predictions.len());

    for (native_value, bridged_value) in native_predictions.iter().zip(bridged_predictions) {
        let native_value = native_value
            .as_f64()
            .expect("native prediction should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged prediction should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-2);
    }

    for key in ["L", "k", "x0"] {
        let native_value = native_result["state"]["parameters"][key]
            .as_f64()
            .expect("native state parameter should be numeric");
        let bridged_value = bridged_result["state"]["parameters"][key]
            .as_f64()
            .expect("bridged state parameter should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-3,
            "{key} mismatch in state parameters: native={native_value}, bridged={bridged_value}"
        );
    }
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(
        native_result["diagnostics"]["uncertainty"]["report_type"],
        "point_estimate"
    );
    let _ = true_params;
}

#[test]
fn native_logistic_prediction_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.predict_model_request("logistic", payload.clone());
    let native = binding
        .predict_model_native(&request)
        .expect("native logistic prediction should succeed");
    let bridged = binding
        .predict_model_via_bridge("logistic", payload)
        .expect("Python bridge prediction should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::PredictModel);
    assert_eq!(native.model_key, bridged.model_key);
    assert_eq!(native.error, None);

    let native_result = native
        .result
        .expect("native response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");
    assert_eq!(native_values.len(), bridged_values.len());

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-12,
            "native value {native_value} should match bridged value {bridged_value}"
        );
    }
}

#[test]
fn native_fisher_pry_fit_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let alpha: f64 = 1.6;
    let t0: f64 = 2.0;
    let observed: Vec<f64> = time
        .iter()
        .map(|t| 1.0 / (1.0 + (-alpha * (t - t0)).exp()))
        .collect();
    let payload = json!({
        "inputs": {
            "time": time,
            "observed": observed,
        }
    });

    let request = binding.fit_model_request("fisher_pry", payload.clone());
    let native = binding
        .fit_model_native(&request)
        .expect("native Fisher-Pry fit should succeed");
    let bridged = binding
        .fit_model_via_bridge("fisher_pry", payload)
        .expect("Python bridge fit should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::FitModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "fisher_pry");
    assert_eq!(native.metadata["runtime"], "rust_native");
    assert_eq!(native.metadata["family"], "substitution");

    let native_result = native.result.expect("native fit should include a result");
    let bridged_result = bridged.result.expect("bridged fit should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);

    for key in ["alpha", "t0"] {
        let native_value = native_result["parameters"][key]
            .as_f64()
            .expect("native parameter should be numeric");
        let bridged_value = bridged_result["parameters"][key]
            .as_f64()
            .expect("bridged parameter should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-2,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }

    let native_predictions = native_result["predictions"]["values"]
        .as_array()
        .expect("native predictions should be an array");
    let bridged_predictions = bridged_result["predictions"]["values"]
        .as_array()
        .expect("bridged predictions should be an array");
    assert_eq!(native_predictions.len(), bridged_predictions.len());

    for (native_value, bridged_value) in native_predictions.iter().zip(bridged_predictions) {
        let native_value = native_value
            .as_f64()
            .expect("native prediction should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged prediction should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-2);
    }

    for key in ["alpha", "t0"] {
        let native_value = native_result["state"]["parameters"][key]
            .as_f64()
            .expect("native state parameter should be numeric");
        let bridged_value = bridged_result["state"]["parameters"][key]
            .as_f64()
            .expect("bridged state parameter should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-2,
            "{key} mismatch in state parameters: native={native_value}, bridged={bridged_value}"
        );
    }
}

#[test]
fn native_fisher_pry_prediction_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        }
    });

    let request = binding.predict_model_request("fisher_pry", payload.clone());
    let native = binding
        .predict_model_native(&request)
        .expect("native Fisher-Pry prediction should succeed");
    let bridged = binding
        .predict_model_via_bridge("fisher_pry", payload)
        .expect("Python bridge prediction should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::PredictModel);
    assert_eq!(native.model_key, bridged.model_key);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "fisher_pry");
    assert_eq!(native.metadata["runtime"], "rust_native");
    assert_eq!(native.metadata["family"], "substitution");

    let native_result = native
        .result
        .expect("native response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");
    assert_eq!(native_values.len(), bridged_values.len());

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-12,
            "native value {native_value} should match bridged value {bridged_value}"
        );
    }
}

#[test]
fn native_fisher_pry_simulation_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
            "time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        }
    });

    let request = binding.simulate_model_request("fisher_pry", payload.clone());
    let native = binding
        .simulate_model_native(&request)
        .expect("native Fisher-Pry simulation should succeed");
    let bridged = binding
        .simulate_model_via_bridge("fisher_pry", payload)
        .expect("Python bridge simulation should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SimulateModel);
    assert_eq!(native.model_key, bridged.model_key);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "fisher_pry");
    assert_eq!(native.metadata["runtime"], "rust_native");
    assert_eq!(native.metadata["family"], "substitution");

    let native_result = native
        .result
        .expect("native response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");
    assert_eq!(native_values.len(), bridged_values.len());

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-12,
            "native value {native_value} should match bridged value {bridged_value}"
        );
    }
}

#[test]
fn native_fisher_pry_summary_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let time: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let observed: Vec<f64> = time
        .iter()
        .map(|t| 1.0 / (1.0 + (-1.6 * (t - 2.0)).exp()))
        .collect();
    let payload = json!({
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
            "time": time,
            "observed": observed
        }
    });

    let request = binding.summarize_model_request("fisher_pry", payload.clone());
    let native = binding
        .summarize_model_native(&request)
        .expect("native Fisher-Pry summary should succeed");

    assert_eq!(native.operation, KernelOperation::SummarizeModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "fisher_pry");
    assert_eq!(native.metadata["runtime"], "rust_native");
    assert_eq!(native.metadata["family"], "substitution");

    let native_result = native
        .result
        .expect("native summary should include a result");

    assert_eq!(native_result["family"], "substitution");
    assert_eq!(native_result["model_name"], "FisherPryModel");
    assert_eq!(native_result["parameter_names"], json!(["alpha", "t0"]));
    assert_eq!(native_result["constructor_kwargs"], json!({}));
    assert_eq!(native_result["state"]["model_key"], "fisher_pry");
    assert_eq!(native_result["state"]["model_name"], "FisherPryModel");
    assert_eq!(native_result["state"]["parameters"]["alpha"], 1.6);
    assert_eq!(native_result["state"]["parameters"]["t0"], 2.0);

    let native_diagnostics = native_result["diagnostics"]
        .as_object()
        .expect("native diagnostics");
    assert_eq!(native_diagnostics["support_level"], "supported");
    assert_eq!(native_diagnostics["provenance"], "deterministic");
    assert_eq!(native_diagnostics["comparison_family"], "fitted");
    assert_eq!(native_diagnostics["model_name"], "FisherPryModel");
    assert_eq!(
        native_diagnostics["uncertainty"]["report_type"],
        "point_estimate"
    );
    assert!(native_diagnostics["warnings"].is_array());
    assert!(native_diagnostics["metrics"].is_object());
}

#[test]
fn native_fisher_pry_diagnose_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let time: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let observed: Vec<f64> = time
        .iter()
        .map(|t| 1.0 / (1.0 + (-1.6 * (t - 2.0)).exp()))
        .collect();
    let payload = json!({
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
            "time": time,
            "observed": observed
        }
    });

    let request = binding.diagnose_model_request("fisher_pry", payload.clone());
    let native = binding
        .diagnose_model_native(&request)
        .expect("native Fisher-Pry diagnostics should succeed");

    assert_eq!(native.operation, KernelOperation::DiagnoseModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "fisher_pry");
    assert_eq!(native.metadata["runtime"], "rust_native");
    assert_eq!(native.metadata["family"], "substitution");

    let native_result = native
        .result
        .expect("native diagnostics should include a result");

    assert_eq!(native_result["state"]["model_key"], "fisher_pry");
    assert_eq!(native_result["state"]["model_name"], "FisherPryModel");
    assert_eq!(native_result["state"]["parameters"]["alpha"], 1.6);
    assert_eq!(native_result["state"]["parameters"]["t0"], 2.0);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(
        native_result["diagnostics"]["uncertainty"]["report_type"],
        "point_estimate"
    );
}

#[test]
fn native_gompertz_fit_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let time = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let observed = vec![0.08, 0.15, 0.25, 0.4, 0.55];
    let payload = json!({
        "inputs": {
            "time": time,
            "observed": observed,
        }
    });

    let request = binding.fit_model_request("gompertz", payload.clone());
    let native = binding
        .fit_model_native(&request)
        .expect("native Gompertz fit should succeed");
    let bridged = binding
        .fit_model_via_bridge("gompertz", payload)
        .expect("Python bridge fit should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::FitModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "gompertz");
    assert_eq!(native.metadata["model_name"], "GompertzModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native.result.expect("native fit should include a result");
    let bridged_result = bridged.result.expect("bridged fit should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);

    let native_predictions = native_result["predictions"]["values"]
        .as_array()
        .expect("native predictions should be an array");
    let bridged_predictions = bridged_result["predictions"]["values"]
        .as_array()
        .expect("bridged predictions should be an array");
    assert_eq!(native_predictions.len(), bridged_predictions.len());

    for (native_value, bridged_value) in native_predictions.iter().zip(bridged_predictions) {
        let native_value = native_value
            .as_f64()
            .expect("native prediction should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged prediction should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 0.05,
            "native prediction {native_value} should match bridged value {bridged_value}"
        );
    }

    let native_parameters = native_result["parameters"]
        .as_object()
        .expect("native parameters should be an object");
    assert!(
        native_parameters["a"]
            .as_f64()
            .expect("a should be numeric")
            > 0.0
    );
    assert!(
        native_parameters["b"]
            .as_f64()
            .expect("b should be numeric")
            > 0.0
    );
    assert!(
        native_parameters["c"]
            .as_f64()
            .expect("c should be numeric")
            > 0.0
    );
}

#[test]
fn native_gompertz_prediction_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.predict_model_request("gompertz", payload.clone());
    let native = binding
        .predict_model_native(&request)
        .expect("native Gompertz prediction should succeed");
    let bridged = binding
        .predict_model_via_bridge("gompertz", payload)
        .expect("Python bridge prediction should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::PredictModel);
    assert_eq!(native.model_key, bridged.model_key);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "gompertz");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");
    assert_eq!(native_values.len(), bridged_values.len());

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!((native_value - bridged_value).abs() < 5e-2);
    }
}

#[test]
fn native_gompertz_simulation_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.simulate_model_request("gompertz", payload.clone());
    let native = binding
        .simulate_model_native(&request)
        .expect("native Gompertz simulation should succeed");
    let bridged = binding
        .simulate_model_via_bridge("gompertz", payload)
        .expect("Python bridge simulation should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SimulateModel);
    assert_eq!(native.model_key, bridged.model_key);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "gompertz");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native response should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged response should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged values should be an array");
    assert_eq!(native_values.len(), bridged_values.len());

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged value should be numeric");
        assert!((native_value - bridged_value).abs() < 5e-2);
    }
}

#[test]
fn native_gompertz_summary_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let time: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let observed: Vec<f64> = time
        .iter()
        .map(|t| 100.0_f64 * (-(3.0_f64 * (-0.3_f64 * t).exp())).exp())
        .collect();
    let payload = json!({
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
            "time": time,
            "observed": observed
        }
    });

    let request = binding.summarize_model_request("gompertz", payload.clone());
    let native = binding
        .summarize_model_native(&request)
        .expect("native Gompertz summary should succeed");
    let bridged = binding
        .summarize_model_via_bridge("gompertz", payload)
        .expect("Python bridge summary should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SummarizeModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "gompertz");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native summary should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged summary should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);
    assert_eq!(native_result["state"]["model_key"], "gompertz");
    assert_eq!(native_result["state"]["model_name"], "GompertzModel");
    assert_eq!(native_result["state"]["parameters"]["a"], 100.0);
    assert_eq!(native_result["state"]["parameters"]["b"], 3.0);
    assert_eq!(native_result["state"]["parameters"]["c"], 0.3);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(native_result["diagnostics"]["model_name"], "GompertzModel");
}

#[test]
fn native_gompertz_diagnose_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let time: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let observed: Vec<f64> = time
        .iter()
        .map(|t| 100.0_f64 * (-(3.0_f64 * (-0.3_f64 * t).exp())).exp())
        .collect();
    let payload = json!({
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
            "time": time,
            "observed": observed
        }
    });

    let request = binding.diagnose_model_request("gompertz", payload.clone());
    let native = binding
        .diagnose_model_native(&request)
        .expect("native Gompertz diagnostics should succeed");
    let bridged = binding
        .diagnose_model_via_bridge("gompertz", payload)
        .expect("Python bridge diagnostics should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::DiagnoseModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "gompertz");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native diagnostics should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged diagnostics should include a result");

    assert_eq!(native_result["state"]["model_key"], "gompertz");
    assert_eq!(native_result["state"]["model_name"], "GompertzModel");
    assert_eq!(native_result["state"]["parameters"]["a"], 100.0);
    assert_eq!(native_result["state"]["parameters"]["b"], 3.0);
    assert_eq!(native_result["state"]["parameters"]["c"], 0.3);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(
        native_result["diagnostics"]["uncertainty"]["report_type"],
        "point_estimate"
    );
    assert_eq!(native_result["diagnostics"]["model_name"], "GompertzModel");
    assert_json_numeric_map_has_finite_values(
        &native_result["diagnostics"]["metrics"],
        &bridged_result["diagnostics"]["metrics"],
    );
}

#[test]
fn native_fit_falls_back_to_bridge_for_non_native_models() {
    let binding = KernelBinding::new();
    let prediction_payload = json!({
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
            "time": [0.0, 1.0, 2.0, 3.0, 4.0]
        }
    });

    let predicted = binding
        .predict_model("bass", prediction_payload)
        .expect("Bass prediction should succeed");
    let observed = predicted
        .result
        .expect("Bass prediction should include a result")["values"]
        .as_array()
        .expect("Bass prediction values should be an array")
        .iter()
        .map(|value| {
            value
                .as_f64()
                .expect("Bass prediction value should be numeric")
        })
        .collect::<Vec<_>>();

    let payload = json!({
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            "observed": observed
        }
    });

    let response = binding
        .fit_model("bass", payload)
        .expect("non-native fit should fall back to the bridge");

    assert_eq!(response.operation, KernelOperation::FitModel);
    assert_eq!(response.metadata["model_key"], "bass");
    assert!(response.result.is_some());
}

#[test]
fn native_logistic_summary_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.summarize_model_request("logistic", payload.clone());
    let native = binding
        .summarize_model_native(&request)
        .expect("native logistic summary should succeed");
    let bridged = binding
        .summarize_model_via_bridge("logistic", payload)
        .expect("Python bridge summary should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SummarizeModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native summary should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged summary should include a result");

    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);
    assert_eq!(
        native_result["parameter_names"],
        bridged_result["parameter_names"]
    );
    assert_eq!(
        native_result["constructor_kwargs"],
        bridged_result["constructor_kwargs"]
    );
    assert_eq!(native_result["state"], bridged_result["state"]);

    let native_diagnostics = native_result["diagnostics"]
        .as_object()
        .expect("native diagnostics");
    let bridged_diagnostics = bridged_result["diagnostics"]
        .as_object()
        .expect("bridged diagnostics");

    assert_eq!(native_diagnostics["support_level"], "supported");
    assert_eq!(native_diagnostics["provenance"], "deterministic");
    assert_eq!(native_diagnostics["comparison_family"], "fitted");
    assert_eq!(native_diagnostics["model_name"], "LogisticModel");
    assert_eq!(
        native_diagnostics["uncertainty"]["report_type"],
        "point_estimate"
    );
    assert_eq!(
        native_diagnostics["warnings"],
        bridged_diagnostics["warnings"]
    );

    for key in ["MSE", "RMSE", "MAE", "R-squared", "R_squared", "RSS"] {
        let native_value = native_diagnostics["metrics"][key]
            .as_f64()
            .expect("native metric should be numeric");
        let bridged_value = bridged_diagnostics["metrics"][key]
            .as_f64()
            .expect("bridged metric should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-6,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }

    let native_residuals = native_diagnostics["residuals"]
        .as_array()
        .expect("native residuals should be an array");
    let bridged_residuals = bridged_diagnostics["residuals"]
        .as_array()
        .expect("bridged residuals should be an array");
    assert_eq!(native_residuals.len(), bridged_residuals.len());
    for (native_value, bridged_value) in native_residuals.iter().zip(bridged_residuals) {
        let native_value = native_value
            .as_f64()
            .expect("native residual should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged residual should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-6);
    }

    let native_analysis = native_diagnostics["residual_analysis"]
        .as_object()
        .expect("native residual analysis should be an object");
    let bridged_analysis = bridged_diagnostics["residual_analysis"]
        .as_object()
        .expect("bridged residual analysis should be an object");
    for key in [
        "mean_residual",
        "max_abs_residual",
        "std_residual",
        "durbin_watson",
    ] {
        let native_value = native_analysis[key]
            .as_f64()
            .expect("native analysis metric");
        let bridged_value = bridged_analysis[key]
            .as_f64()
            .expect("bridged analysis metric");
        assert!(
            (native_value - bridged_value).abs() < 1e-6,
            "{key} mismatch"
        );
    }
}

#[test]
fn native_logistic_diagnose_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.diagnose_model_request("logistic", payload.clone());
    let native = binding
        .diagnose_model_native(&request)
        .expect("native logistic diagnostics should succeed");
    let bridged = binding
        .diagnose_model_via_bridge("logistic", payload)
        .expect("Python bridge diagnostics should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::DiagnoseModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "logistic");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native diagnostics should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged diagnostics should include a result");

    assert_eq!(native_result["state"], bridged_result["state"]);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
    assert_eq!(native_result["diagnostics"]["provenance"], "deterministic");
    assert_eq!(native_result["diagnostics"]["comparison_family"], "fitted");
    assert_eq!(
        native_result["diagnostics"]["uncertainty"]["report_type"],
        "point_estimate"
    );

    for key in ["MSE", "RMSE", "MAE", "R-squared", "R_squared", "RSS"] {
        let native_value = native_result["diagnostics"]["metrics"][key]
            .as_f64()
            .expect("native metric should be numeric");
        let bridged_value = bridged_result["diagnostics"]["metrics"][key]
            .as_f64()
            .expect("bridged metric should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-6,
            "{key} mismatch: native={native_value}, bridged={bridged_value}"
        );
    }
}

#[test]
fn native_summary_and_diagnose_reject_unsupported_native_payloads() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "logistic",
            "model_name": "LogisticModel",
            "constructor_kwargs": {
                "covariates": ["marketing_spend"]
            },
            "parameters": {
                "L": 100.0,
                "k": 0.65,
                "x0": 3.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [12.45533581839509, 21.416501696862853, 34.29895373044305, 50.0],
            "covariates": {
                "marketing_spend": [0.2, 0.4, 0.6, 0.8]
            }
        }
    });

    let summary_error = binding
        .summarize_model("logistic", payload.clone())
        .expect_err("native summary should reject unsupported covariate payloads");
    let diagnose_error = binding
        .diagnose_model("logistic", payload)
        .expect_err("native diagnostics should reject unsupported covariate payloads");

    assert_eq!(summary_error.code, "unsupported_native_operation");
    assert_eq!(
        summary_error.operation,
        Some(KernelOperation::SummarizeModel)
    );
    assert!(summary_error.message.contains("logistic"));
    assert_eq!(diagnose_error.code, "unsupported_native_operation");
    assert_eq!(
        diagnose_error.operation,
        Some(KernelOperation::DiagnoseModel)
    );
    assert!(diagnose_error.message.contains("logistic"));
}

#[test]
fn native_norton_bass_prediction_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = norton_bass_prediction_payload(1, vec![0.0, 1.0, 2.0, 3.0]);

    let request = binding.predict_model_request("norton_bass", payload.clone());
    let native = binding
        .predict_model_native(&request)
        .expect("native Norton-Bass prediction should succeed");
    let bridged = binding
        .predict_model_via_bridge("norton_bass", payload)
        .expect("Python bridge Norton-Bass prediction should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::PredictModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "norton_bass");
    assert_eq!(native.metadata["model_name"], "NortonBassModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native Norton-Bass prediction should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Norton-Bass prediction should include a result");

    assert_eq!(native_result["columns"], bridged_result["columns"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_rows = native_result["rows"]
        .as_array()
        .expect("native Norton-Bass rows should be an array");
    let bridged_rows = bridged_result["rows"]
        .as_array()
        .expect("bridged Norton-Bass rows should be an array");
    assert_eq!(native_rows.len(), bridged_rows.len());
    assert_eq!(native_rows.len(), 4);

    for (native_row, bridged_row) in native_rows.iter().zip(bridged_rows) {
        let native_value = native_row
            .as_array()
            .and_then(|row| row.first())
            .and_then(|value| value.as_f64())
            .expect("native Norton-Bass prediction should be numeric");
        let bridged_value = bridged_row
            .as_array()
            .and_then(|row| row.first())
            .and_then(|value| value.as_f64())
            .expect("bridged Norton-Bass prediction should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-5);
    }
}

#[test]
fn native_norton_bass_simulation_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = norton_bass_prediction_payload(1, vec![0.0, 1.0, 2.0, 3.0]);

    let request = binding.simulate_model_request("norton_bass", payload.clone());
    let native = binding
        .simulate_model_native(&request)
        .expect("native Norton-Bass simulation should succeed");
    let bridged = binding
        .simulate_model_via_bridge("norton_bass", payload)
        .expect("Python bridge Norton-Bass simulation should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SimulateModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "norton_bass");
    assert_eq!(native.metadata["model_name"], "NortonBassModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native Norton-Bass simulation should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Norton-Bass simulation should include a result");

    assert_eq!(native_result["columns"], bridged_result["columns"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_rows = native_result["rows"]
        .as_array()
        .expect("native Norton-Bass rows should be an array");
    let bridged_rows = bridged_result["rows"]
        .as_array()
        .expect("bridged Norton-Bass rows should be an array");
    assert_eq!(native_rows.len(), bridged_rows.len());
    assert_eq!(native_rows.len(), 4);

    for (native_row, bridged_row) in native_rows.iter().zip(bridged_rows) {
        let native_value = native_row
            .as_array()
            .and_then(|row| row.first())
            .and_then(|value| value.as_f64())
            .expect("native Norton-Bass value should be numeric");
        let bridged_value = bridged_row
            .as_array()
            .and_then(|row| row.first())
            .and_then(|value| value.as_f64())
            .expect("bridged Norton-Bass value should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-5);
    }
}

#[test]
fn native_norton_bass_fallback_paths_emit_tracing_events_for_unsupported_payloads() {
    let binding = KernelBinding::new();
    let payload = norton_bass_prediction_payload(1, vec![1.0, 2.0, 3.0, 4.0]);

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(SharedBufferWriter(buffer.clone()))
        .without_time()
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let response = binding
        .predict_model("norton_bass", payload)
        .expect("unsupported Norton-Bass prediction should fall back to the Python bridge");

    let logs = String::from_utf8(buffer.lock().expect("buffer should be readable").clone())
        .expect("tracing output should be valid UTF-8");

    assert_eq!(response.operation, KernelOperation::PredictModel);
    assert!(logs.contains("native path unsupported, falling back to Python bridge"));
    assert!(logs.contains("norton_bass"));
}

#[test]
fn native_norton_bass_simulation_falls_back_to_bridge_for_unsupported_payloads() {
    let binding = KernelBinding::new();
    let payload = norton_bass_prediction_payload(1, vec![1.0, 2.0, 3.0, 4.0]);

    let response = binding
        .simulate_model("norton_bass", payload)
        .expect("unsupported Norton-Bass simulation should fall back to the Python bridge");

    assert_eq!(response.operation, KernelOperation::SimulateModel);
    assert_eq!(response.metadata["model_key"], "norton_bass");
    assert!(response.result.is_some());
}

#[test]
fn native_norton_bass_summary_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = norton_bass_analysis_payload(1);

    let request = binding.summarize_model_request("norton_bass", payload.clone());
    let native = binding
        .summarize_model_native(&request)
        .expect("native Norton-Bass summary should succeed");
    let bridged = binding
        .summarize_model("norton_bass", payload)
        .expect("public Norton-Bass summary should use the native path");

    assert_eq!(native, bridged);
    assert_eq!(native.operation, KernelOperation::SummarizeModel);
    assert_eq!(native.metadata["model_key"], "norton_bass");
    assert_eq!(native.metadata["model_name"], "NortonBassModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let summary = native
        .diagnostics_summary()
        .expect("native Norton-Bass summary should expose diagnostics summary");
    assert_eq!(summary.support_level, "supported");
    assert_eq!(summary.provenance, "deterministic");
    assert_eq!(summary.comparison_family, "fitted");
    assert_eq!(summary.model_name.as_deref(), Some("NortonBassModel"));

    let native_result = native
        .result
        .expect("native Norton-Bass summary should include a result");
    assert_eq!(native_result["model_name"], "NortonBassModel");
    assert_eq!(native_result["parameter_names"], json!(["p1", "q1", "m1"]));
    assert_eq!(native_result["state"]["model_name"], "NortonBassModel");
    assert_eq!(native_result["state"]["parameters"]["p1"], 0.001);
    assert_eq!(native_result["state"]["parameters"]["q1"], 0.1);
    assert_eq!(native_result["state"]["parameters"]["m1"], 100.0);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
}

#[test]
fn native_norton_bass_diagnose_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = norton_bass_analysis_payload(1);

    let request = binding.diagnose_model_request("norton_bass", payload.clone());
    let native = binding
        .diagnose_model_native(&request)
        .expect("native Norton-Bass diagnostics should succeed");
    let bridged = binding
        .diagnose_model("norton_bass", payload)
        .expect("public Norton-Bass diagnostics should use the native path");

    assert_eq!(native, bridged);
    assert_eq!(native.operation, KernelOperation::DiagnoseModel);
    assert_eq!(native.metadata["model_key"], "norton_bass");
    assert_eq!(native.metadata["model_name"], "NortonBassModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let summary = native
        .diagnostics_summary()
        .expect("native Norton-Bass diagnostics should expose diagnostics summary");
    assert_eq!(summary.support_level, "supported");
    assert_eq!(summary.provenance, "deterministic");
    assert_eq!(summary.comparison_family, "fitted");
    assert_eq!(summary.model_name.as_deref(), Some("NortonBassModel"));

    let native_result = native
        .result
        .expect("native Norton-Bass diagnostics should include a result");
    assert_eq!(native_result["state"]["model_name"], "NortonBassModel");
    assert_eq!(native_result["state"]["parameters"]["p1"], 0.001);
    assert_eq!(native_result["state"]["parameters"]["q1"], 0.1);
    assert_eq!(native_result["state"]["parameters"]["m1"], 100.0);
    assert_eq!(native_result["diagnostics"]["support_level"], "supported");
}

#[test]
fn native_bass_fit_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = bass_fit_payload();

    let request = binding.fit_model_request("bass", payload.clone());
    let native = binding
        .fit_model_native(&request)
        .expect("native Bass fit should succeed");
    let bridged = binding
        .fit_model_via_bridge("bass", payload)
        .expect("Python bridge Bass fit should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::FitModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "bass");
    assert_eq!(native.metadata["model_name"], "BassModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_summary = native
        .diagnostics_summary()
        .expect("native Bass fit should expose diagnostics summary");
    let bridged_summary = bridged
        .diagnostics_summary()
        .expect("bridged Bass fit should expose diagnostics summary");
    assert_eq!(native_summary, bridged_summary);
    assert_eq!(native_summary.support_level, "supported");
    assert_eq!(native_summary.provenance, "deterministic");
    assert_eq!(native_summary.comparison_family, "fitted");
    assert_eq!(native_summary.model_name.as_deref(), Some("BassModel"));

    let native_result = native
        .result
        .expect("native Bass fit should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Bass fit should include a result");
    assert_eq!(native_result["family"], bridged_result["family"]);
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);
    for key in ["p", "q", "m"] {
        let native_value = native_result["parameters"][key]
            .as_f64()
            .expect("native Bass parameter should be numeric");
        let bridged_value = bridged_result["parameters"][key]
            .as_f64()
            .expect("bridged Bass parameter should be numeric");
        assert!(native_value.is_finite() && bridged_value.is_finite());
        assert!(
            native_value > 0.0,
            "native Bass parameter '{key}' should be positive"
        );
        assert!(
            bridged_value > 0.0,
            "bridged Bass parameter '{key}' should be positive"
        );
    }

    let native_predictions = native_result["predictions"]["values"]
        .as_array()
        .expect("native Bass predictions should be an array");
    let bridged_predictions = bridged_result["predictions"]["values"]
        .as_array()
        .expect("bridged Bass predictions should be an array");
    assert_eq!(native_predictions.len(), bridged_predictions.len());
    assert_eq!(
        native_result["predictions"]["shape"],
        bridged_result["predictions"]["shape"]
    );
    assert_eq!(
        native_result["predictions"]["dtype"],
        bridged_result["predictions"]["dtype"]
    );
    assert_eq!(
        native_result["predictions"]["metadata"],
        bridged_result["predictions"]["metadata"]
    );

    for native_value in native_predictions {
        let native_value = native_value
            .as_f64()
            .expect("native Bass prediction should be numeric");
        assert!(native_value.is_finite());
    }
}

#[test]
fn native_bass_prediction_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.predict_model_request("bass", payload.clone());
    let native = binding
        .predict_model_native(&request)
        .expect("native Bass prediction should succeed");
    let bridged = binding
        .predict_model_via_bridge("bass", payload)
        .expect("Python bridge Bass prediction should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::PredictModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "bass");
    assert_eq!(native.metadata["model_name"], "BassModel");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native Bass prediction should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Bass prediction should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);
    assert_eq!(native_result["metadata"], bridged_result["metadata"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native Bass values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged Bass values should be an array");

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native Bass value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged Bass value should be numeric");
        assert!(
            (native_value - bridged_value).abs() < 1e-5,
            "native Bass value {native_value} should match bridged value {bridged_value}"
        );
    }
}

#[test]
fn native_bass_simulation_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = json!({
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
    });

    let request = binding.simulate_model_request("bass", payload.clone());
    let native = binding
        .simulate_model_native(&request)
        .expect("native Bass simulation should succeed");
    let bridged = binding
        .simulate_model_via_bridge("bass", payload)
        .expect("Python bridge Bass simulation should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SimulateModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "bass");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_result = native
        .result
        .expect("native Bass simulation should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Bass simulation should include a result");

    assert_eq!(native_result["shape"], bridged_result["shape"]);
    assert_eq!(native_result["dtype"], bridged_result["dtype"]);

    let native_values = native_result["values"]
        .as_array()
        .expect("native Bass values should be an array");
    let bridged_values = bridged_result["values"]
        .as_array()
        .expect("bridged Bass values should be an array");

    for (native_value, bridged_value) in native_values.iter().zip(bridged_values) {
        let native_value = native_value
            .as_f64()
            .expect("native Bass value should be numeric");
        let bridged_value = bridged_value
            .as_f64()
            .expect("bridged Bass value should be numeric");
        assert!((native_value - bridged_value).abs() < 1e-5);
    }
}

#[test]
fn native_bass_summary_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = bass_analysis_payload();

    let request = binding.summarize_model_request("bass", payload.clone());
    let native = binding
        .summarize_model_native(&request)
        .expect("native Bass summary should succeed");
    let bridged = binding
        .summarize_model_via_bridge("bass", payload)
        .expect("Python bridge Bass summary should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::SummarizeModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "bass");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_summary = native
        .diagnostics_summary()
        .expect("native Bass summary should expose diagnostics summary");
    let bridged_summary = bridged
        .diagnostics_summary()
        .expect("bridged Bass summary should expose diagnostics summary");
    assert_eq!(native_summary, bridged_summary);
    assert_eq!(native_summary.support_level, "supported");
    assert_eq!(native_summary.provenance, "deterministic");
    assert_eq!(native_summary.comparison_family, "fitted");
    assert_eq!(native_summary.model_name.as_deref(), Some("BassModel"));

    let native_result = native
        .result
        .expect("native Bass summary should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Bass summary should include a result");
    assert_eq!(native_result["model_name"], bridged_result["model_name"]);
    assert_eq!(native_result["state"], bridged_result["state"]);
    assert_json_numeric_map_has_finite_values(
        &native_result["diagnostics"]["metrics"],
        &bridged_result["diagnostics"]["metrics"],
    );
}

#[test]
fn native_bass_diagnose_matches_python_bridge_contract() {
    let binding = KernelBinding::new();
    let payload = bass_analysis_payload();

    let request = binding.diagnose_model_request("bass", payload.clone());
    let native = binding
        .diagnose_model_native(&request)
        .expect("native Bass diagnostics should succeed");
    let bridged = binding
        .diagnose_model_via_bridge("bass", payload)
        .expect("Python bridge Bass diagnostics should succeed");

    assert_eq!(native.schema_version, bridged.schema_version);
    assert_eq!(native.operation, KernelOperation::DiagnoseModel);
    assert_eq!(native.error, None);
    assert_eq!(native.metadata["model_key"], "bass");
    assert_eq!(native.metadata["runtime"], "rust_native");

    let native_summary = native
        .diagnostics_summary()
        .expect("native Bass diagnostics should expose diagnostics summary");
    let bridged_summary = bridged
        .diagnostics_summary()
        .expect("bridged Bass diagnostics should expose diagnostics summary");
    assert_eq!(native_summary, bridged_summary);
    assert_eq!(native_summary.support_level, "supported");
    assert_eq!(native_summary.provenance, "deterministic");
    assert_eq!(native_summary.comparison_family, "fitted");
    assert_eq!(native_summary.model_name.as_deref(), Some("BassModel"));

    let native_result = native
        .result
        .expect("native Bass diagnostics should include a result");
    let bridged_result = bridged
        .result
        .expect("bridged Bass diagnostics should include a result");
    assert_eq!(native_result["state"], bridged_result["state"]);
    assert_json_numeric_map_has_finite_values(
        &native_result["diagnostics"]["metrics"],
        &bridged_result["diagnostics"]["metrics"],
    );
}

#[test]
fn native_bass_reports_structured_errors_for_invalid_or_unsupported_shapes() {
    let binding = KernelBinding::new();
    let missing_parameter_payload = json!({
        "state": {
            "model_key": "bass",
            "model_name": "BassModel",
            "constructor_kwargs": {},
            "parameters": {
                "p": 0.03,
                "m": 120.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0]
        }
    });
    let invalid_request = binding.predict_model_request("bass", missing_parameter_payload);
    let invalid_error = binding
        .predict_model_native(&invalid_request)
        .expect_err("missing Bass q parameter should be a structured native error");
    assert_eq!(invalid_error.code, "invalid_request");
    assert_eq!(invalid_error.operation, Some(KernelOperation::PredictModel));
    assert!(invalid_error
        .message
        .contains("missing numeric parameter 'q'"));

    let covariate_payload = json!({
        "state": {
            "model_key": "bass",
            "model_name": "BassModel",
            "constructor_kwargs": {
                "covariates": ["price"]
            },
            "parameters": {
                "p": 0.03,
                "q": 0.38,
                "m": 120.0,
                "beta_p_price": 0.0,
                "beta_q_price": 0.0,
                "beta_m_price": 0.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0],
            "covariates": {
                "price": [1.0, 1.1, 1.2]
            }
        }
    });
    let unsupported_request = binding.predict_model_request("bass", covariate_payload);
    let unsupported_error = binding
        .predict_model_native(&unsupported_request)
        .expect_err("Bass covariates should remain on the bridge path");
    assert_eq!(unsupported_error.code, "unsupported_native_operation");
    assert_eq!(
        unsupported_error.operation,
        Some(KernelOperation::PredictModel)
    );
    assert!(unsupported_error.message.contains("without covariates"));

    let shifted_time_payload = json!({
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
            "time": [1.0, 2.0, 3.0]
        }
    });
    let shifted_time_request = binding.predict_model_request("bass", shifted_time_payload);
    let shifted_time_error = binding
        .predict_model_native(&shifted_time_request)
        .expect_err("Bass time grids with non-zero starts should remain on the bridge path");
    assert_eq!(shifted_time_error.code, "unsupported_native_operation");
    assert!(shifted_time_error.message.contains("start at zero"));
}

#[test]
fn native_prediction_falls_back_to_bridge_for_non_native_models() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "norton_bass",
            "model_name": "NortonBassModel",
            "constructor_kwargs": {},
            "parameters": {
                "p1": 0.001,
                "q1": 0.1,
                "m1": 100.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    });

    let response = binding
        .predict_model("norton_bass", payload)
        .expect("non-native prediction should fall back to the bridge");

    assert_eq!(response.operation, KernelOperation::PredictModel);
    assert_eq!(response.metadata["model_key"], "norton_bass");
    assert!(response.result.is_some());
}

#[test]
fn native_simulation_falls_back_to_bridge_for_non_native_models() {
    let binding = KernelBinding::new();
    let payload = json!({
        "state": {
            "model_key": "norton_bass",
            "model_name": "NortonBassModel",
            "constructor_kwargs": {},
            "parameters": {
                "p1": 0.001,
                "q1": 0.1,
                "m1": 100.0
            },
            "predict_kwargs": {}
        },
        "inputs": {
            "time": [0.0, 1.0, 2.0, 3.0],
            "observed": [0.05, 0.12, 0.3, 0.6]
        }
    });

    let response = binding
        .simulate_model("norton_bass", payload)
        .expect("non-native simulation should fall back to the bridge");

    assert_eq!(response.operation, KernelOperation::SimulateModel);
    assert_eq!(response.metadata["model_key"], "norton_bass");
    assert!(response.result.is_some());
}
