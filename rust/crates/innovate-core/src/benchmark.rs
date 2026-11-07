//! Reproducible synthetic smoke corpus and native fit/forecast runner.
use serde_json::{Value, json};

const CASES: [(&str, &str, &str); 4] = [
    ("bass_smoke_adoption", "bass", "diffusion"),
    ("logistic_growth_smoke", "logistic", "diffusion"),
    ("fisher_pry_replacement_smoke", "fisher_pry", "substitution"),
    (
        "lotka_volterra_competition_smoke",
        "multi_product",
        "competition",
    ),
];

fn case_descriptor(id: &str, model: &str, family: &str) -> Value {
    json!({"case_id":id,"canonical_model_key":model,"family":family,
        "dataset_version":"2026.04","source":"synthetic"})
}

fn discover() -> Value {
    let cases: Vec<_> = CASES
        .iter()
        .map(|&(id, model, family)| case_descriptor(id, model, family))
        .collect();
    json!({"cases":cases})
}

fn run(payload: &Value) -> Result<Value, String> {
    let id = payload["case_id"]
        .as_str()
        .ok_or("benchmark requires case_id")?;
    run_case(id)
}

fn run_case(id: &str) -> Result<Value, String> {
    let &(id, model, family) = CASES
        .iter()
        .find(|case| case.0 == id)
        .ok_or("unknown benchmark case")?;
    let time: Vec<f64> = (0..12).map(f64::from).collect();
    let sigmoid = |t: f64, rate: f64, center: f64| 1. / (1. + (-rate * (t - center)).exp());
    let observed: Vec<Value> = time
        .iter()
        .map(|&t| match model {
            "bass" => {
                let decay = (-0.41 * t).exp();
                json!(1000. * (1. - decay) / (1. + (0.38 / 0.03) * decay))
            }
            "logistic" => json!(1000. * sigmoid(t, 0.8, 6.)),
            "fisher_pry" => json!(sigmoid(t, 1., 6.)),
            _ => json!([700. * sigmoid(t, 0.65, 5.), 300. * sigmoid(t, 0.45, 6.5)]),
        })
        .collect();
    let kernel = crate::KernelBinding::new();
    let fit = kernel
        .fit_model(model, json!({"inputs":{"time":time,"observed":observed}}))
        .map_err(|error| format!("benchmark fit failed: {}", error.message))?
        .result
        .ok_or("benchmark fit returned no result")?;
    let prediction = kernel
        .predict_model(model, json!({"inputs":{"time":time},"state":fit["state"]}))
        .map_err(|error| format!("benchmark prediction failed: {}", error.message))?
        .result
        .ok_or("benchmark prediction returned no result")?;
    Ok(
        json!({"case":case_descriptor(id,model,family),"time":time,"observed":observed,
        "fit":fit,"prediction":prediction,"scope":"in_sample_synthetic_smoke"}),
    )
}

fn suite(payload: &Value) -> Result<Value, String> {
    let workers = crate::batch_workers(payload).ok_or("workers must be an integer in 1..=32")?;
    let ids: Vec<&str> = match payload.get("case_ids") {
        None => CASES.iter().map(|case| case.0).collect(),
        Some(value) => {
            let entries = value
                .as_array()
                .filter(|entries| !entries.is_empty() && entries.len() <= CASES.len())
                .ok_or("case_ids must contain one to four case IDs")?;
            entries
                .iter()
                .map(|entry| {
                    entry
                        .as_str()
                        .filter(|id| !id.trim().is_empty())
                        .ok_or("case IDs must be nonblank strings")
                })
                .collect::<Result<_, _>>()?
        }
    };
    if ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len() {
        return Err("benchmark case IDs must be unique".into());
    }
    let error_row = |id: &str, code: &str, message: &str| json!({"case_id":id,"result":null,"error":{"code":code,"message":message}});
    let runs = crate::run_ordered_chunks(
        &ids,
        workers,
        |chunk| {
            chunk
                .iter()
                .map(|id| match run_case(id) {
                    Ok(result) => json!({"case_id":id,"result":result,"error":null}),
                    Err(message) => error_row(id, "benchmark_case_failed", &message),
                })
                .collect()
        },
        |id| {
            error_row(
                id,
                "batch_worker_panicked",
                "native benchmark worker failed",
            )
        },
    );
    let succeeded = runs.iter().filter(|row| row["error"].is_null()).count();
    Ok(
        json!({"runs":runs,"succeeded":succeeded,"failed":runs.len()-succeeded,
        "scope":"in_sample_synthetic_smoke"}),
    )
}

pub(crate) fn execute(request: &Value) -> Value {
    let operation = &request["operation"];
    let result = if request["schema_version"] != crate::KERNEL_SCHEMA_VERSION {
        Err("benchmark request requires schema 2.0".into())
    } else if operation == "discover_benchmark_cases" {
        Ok(discover())
    } else if operation == "run_benchmark_suite" {
        suite(&request["payload"])
    } else {
        run(&request["payload"])
    };
    let (result, error) = match result {
        Ok(result) => (result, Value::Null),
        Err(message) => (
            Value::Null,
            json!({"code":"invalid_request","message":message}),
        ),
    };
    json!({"schema_version":crate::KERNEL_SCHEMA_VERSION,"operation":operation,
        "result":result,"error":error,"metadata":{"runtime":"rust_native"}})
}
