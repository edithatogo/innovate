//! Reproducible synthetic smoke corpus and native fit/forecast runner.
use serde_json::{Value, json};

const CASES: [(&str, &str, &str); 6] = [
    ("bass_smoke_adoption", "bass", "diffusion"),
    ("logistic_growth_smoke", "logistic", "diffusion"),
    ("fisher_pry_replacement_smoke", "fisher_pry", "substitution"),
    (
        "lotka_volterra_competition_smoke",
        "multi_product",
        "competition",
    ),
    ("lock_in_smoke", "lock_in", "competition"),
    (
        "network_diffusion_smoke",
        "network_diffusion",
        "network_diffusion",
    ),
];

fn ci_policy(family: &str) -> &'static str {
    match family {
        "competition" => "scheduled",
        "network_diffusion" => "workflow_dispatch",
        _ => "fast",
    }
}

fn case_descriptor(id: &str, model: &str, family: &str) -> Value {
    json!({"case_id":id,"canonical_model_key":model,"family":family,
        "dataset_version":match model {"lock_in"=>"native-lock-in-1","network_diffusion"=>"native-network-1",_=>"2026.04"},"source":"synthetic","ci_policy":ci_policy(family)})
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
    if model == "network_diffusion" {
        let time: Vec<f64> = (0..26).map(f64::from).collect();
        let bass = |t: f64, p: f64, q: f64, m: f64| {
            let decay = (-(p + q) * t).exp();
            m * (1. - decay) / (1. + (q / p) * decay)
        };
        let observed: Vec<_> = time
            .iter()
            .map(|&t| [bass(t, 0.03, 0.3, 100.), bass(t, 0.02, 0.2, 200.)])
            .collect();
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time,"observed":observed},
                "constructor_kwargs":{"adjacency":[[0.,1.],[1.,0.]],"orientation":"time_node","spillover_strength":0.}
            }),
        );
    }
    if model == "lock_in" {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/lock_in_scipy.json"))
                .expect("embedded independent lock-in reference");
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":fixture["time"],"observed":fixture["rows"]},
                "fit_options":{"initial_parameters":fixture["parameters"]}
            }),
        );
    }
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
    fit_and_forecast(
        id,
        model,
        family,
        json!({"inputs":{"time":time,"observed":observed}}),
    )
}

fn fit_and_forecast(id: &str, model: &str, family: &str, payload: Value) -> Result<Value, String> {
    let time = payload["inputs"]["time"].clone();
    let observed = payload["inputs"]["observed"].clone();
    let kernel = crate::KernelBinding::new();
    let fit = kernel
        .fit_model(model, payload)
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
    let policy = match payload.get("ci_policy") {
        None => None,
        Some(value) => Some(
            value
                .as_str()
                .filter(|policy| matches!(*policy, "fast" | "scheduled" | "workflow_dispatch"))
                .ok_or("ci_policy must be fast, scheduled or workflow_dispatch")?,
        ),
    };
    if policy.is_some() && payload.get("case_ids").is_some() {
        return Err("select either case_ids or ci_policy".into());
    }
    let ids: Vec<&str> = match payload.get("case_ids") {
        None => CASES
            .iter()
            .filter(|case| policy.is_none_or(|policy| ci_policy(case.2) == policy))
            .map(|case| case.0)
            .collect(),
        Some(value) => {
            let entries = value
                .as_array()
                .filter(|entries| !entries.is_empty() && entries.len() <= CASES.len())
                .ok_or("case_ids must contain one to six case IDs")?;
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

fn validate_corpus() -> Result<Value, String> {
    let cards = crate::model_cards::discover(&json!({}))?;
    let cards = cards["cards"]
        .as_array()
        .expect("model card discovery returns an array");
    let case_versions: std::collections::BTreeMap<_, _> = CASES
        .iter()
        .map(|&(id, model, family)| {
            (
                id,
                case_descriptor(id, model, family)["dataset_version"].clone(),
            )
        })
        .collect();
    let mut issues = Vec::new();
    let mut summaries = serde_json::Map::new();
    for card in cards {
        let key = card["model_key"].as_str().expect("discovered model key");
        let Some(identifiers) = card["historical_benchmark_card"]["benchmark_case_ids"]
            .as_array()
            .or_else(|| card["native_model_card"]["benchmark_case_ids"].as_array())
            .filter(|ids| !ids.is_empty())
        else {
            continue;
        };
        let mut versions = serde_json::Map::new();
        for id in identifiers {
            let id = id.as_str().expect("embedded benchmark ID");
            if let Some(version) = case_versions.get(id) {
                versions.insert(id.to_owned(), version.clone());
            } else {
                issues.push(json!({"code":"unknown_benchmark_case","scope":"model_card",
                    "identifier":key,"severity":"error","message":format!("Missing benchmark case '{id}'.")}));
            }
        }
        let status = if versions.len() == identifiers.len() {
            "current"
        } else {
            "stale"
        };
        summaries.insert(key.to_owned(), json!({"model_key":key,"benchmark_case_ids":identifiers,
            "benchmark_case_versions":versions,"supported_backends":card["native_capability"]["supported_backends"],
            "freshness":{"status":status,"rule":"all referenced benchmark cases must exist with dataset versions"}}));
    }
    Ok(json!({"ok":issues.is_empty(),"issues":issues,
        "summary":{"case_count":case_versions.len(),"model_card_count":summaries.len(),
            "fast_ci_case_count":CASES.iter().filter(|case| ci_policy(case.2) == "fast").count(),
            "scheduled_case_count":CASES.iter().filter(|case| ci_policy(case.2) == "scheduled").count(),
            "manual_case_count":CASES.iter().filter(|case| ci_policy(case.2) == "workflow_dispatch").count()},
        "model_card_summaries":summaries}))
}

pub(crate) fn execute(request: &Value) -> Value {
    let operation = &request["operation"];
    let result = if request["schema_version"] != crate::KERNEL_SCHEMA_VERSION {
        Err("benchmark request requires schema 2.0".into())
    } else if operation == "discover_benchmark_cases" {
        Ok(discover())
    } else if operation == "validate_benchmark_corpus" {
        validate_corpus()
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
