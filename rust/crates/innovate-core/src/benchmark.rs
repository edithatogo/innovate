//! Reproducible synthetic smoke corpus and native fit/forecast runner.
use serde_json::{Value, json};

const CASES: [(&str, &str, &str); 16] = [
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
    (
        "network_spillover_reference",
        "network_diffusion",
        "network_diffusion",
    ),
    ("policy_hazard_reference", "policy_hazard", "policy_hazard"),
    ("policy_hazard_fit", "policy_hazard", "policy_hazard"),
    (
        "latent_process_reference",
        "latent_process",
        "latent_process",
    ),
    ("latent_process_fit", "latent_process", "latent_process"),
    ("regime_switching_reference", "regime_switching", "regime"),
    ("regime_switching_fit", "regime_switching", "regime"),
    ("hierarchical_group_fit", "hierarchical", "hierarchical"),
    ("mixture_reference", "mixture", "mixture"),
    ("mixture_gaussian_fit", "mixture", "mixture"),
];

fn ci_policy(family: &str) -> &'static str {
    match family {
        "competition" => "scheduled",
        "network_diffusion" | "policy_hazard" | "latent_process" | "regime" | "hierarchical"
        | "mixture" => "workflow_dispatch",
        _ => "fast",
    }
}

fn case_descriptor(id: &str, model: &str, family: &str) -> Value {
    json!({"case_id":id,"canonical_model_key":model,"family":family,
        "dataset_version":match model {"lock_in"=>"native-lock-in-1","network_diffusion"=>"native-network-1","policy_hazard"=>"native-policy-1","latent_process"=>"native-latent-1","regime_switching"=>"native-regime-1","hierarchical"=>"native-hierarchical-1","mixture"=>"native-mixture-1",_=>"2026.04"},"source":"synthetic","ci_policy":ci_policy(family)})
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
    if id == "mixture_gaussian_fit" {
        let time: Vec<_> = (0..50).map(|i| f64::from(i) * 0.5).collect();
        let observed: Vec<_> = time
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if i % 2 == 0 {
                    100. / (1. + (-0.4 * (t - 6.)).exp())
                } else {
                    200. / (1. + (-0.3 * (t - 10.)).exp())
                }
            })
            .collect();
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time,"observed":observed},
                "constructor_kwargs":{"components":["logistic","logistic"],"max_iter":30,"noise_floor":0.001},
                "fit_options":{"initial_parameters":{"model_0_L":98.,"model_0_k":0.4,"model_0_x0":6.,
                    "model_1_L":202.,"model_1_k":0.3,"model_1_x0":10.,
                    "weight_0":0.4,"weight_1":0.6,"noise_scale":2.}}
            }),
        );
    }
    if id == "mixture_reference" {
        let time = [0_f64, 4., 6., 10., 16.];
        let observed: Vec<_> = time
            .iter()
            .map(|&t| 40. / (1. + (-0.4 * (t - 6.)).exp()) + 120. / (1. + (-0.3 * (t - 10.)).exp()))
            .collect();
        return fixed_prediction(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time},
                "constructor_kwargs":{"components":["logistic","logistic"]},
                "parameters":{"model_0_L":100.,"model_0_k":0.4,"model_0_x0":6.,
                    "model_1_L":200.,"model_1_k":0.3,"model_1_x0":10.,
                    "weight_0":0.4,"weight_1":0.6,"noise_scale":2.}
            }),
            json!(observed),
        );
    }
    if id == "hierarchical_group_fit" {
        let time: Vec<f64> = (0..25).map(f64::from).collect();
        let observed: Vec<_> = time
            .iter()
            .map(|&t| {
                [
                    100. / (1. + (-0.4 * (t - 7.)).exp()),
                    200. / (1. + (-0.3 * (t - 9.)).exp()),
                ]
            })
            .collect();
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time,"observed":observed},
                "constructor_kwargs":{"base_model":"logistic","groups":["a","b"],"output":"groups"}
            }),
        );
    }
    if id == "regime_switching_fit" {
        let time: Vec<f64> = (0..30).map(f64::from).collect();
        let observed: Vec<_> = time
            .iter()
            .map(|&t| 100. / (1. + (-0.4 * (t - 8.)).exp()))
            .collect();
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time,"observed":observed},
                "constructor_kwargs":{"base_model":"logistic","prediction_mode":"fixed_time"}
            }),
        );
    }
    if id == "regime_switching_reference" {
        let time = [0_f64, 1., 2.5, 4., 7.];
        let observed: Vec<_> = time
            .iter()
            .map(|&t| {
                if t < 2.5 {
                    100. / (1. + (2.5 - t).exp())
                } else {
                    200. / (1. + (0.5 * (2.5 - t)).exp()) - 50.
                }
            })
            .collect();
        return fixed_prediction(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time},
                "constructor_kwargs":{"base_model":"logistic","prediction_mode":"fixed_time","split_time":2.5},
                "parameters":{"changepoint_index":2,"regime_0_L":100.,"regime_0_k":1.,"regime_0_x0":2.5,
                    "regime_1_L":200.,"regime_1_k":0.5,"regime_1_x0":2.5}
            }),
            json!(observed),
        );
    }
    if id == "latent_process_fit" {
        let time: Vec<f64> = (0..21).map(f64::from).collect();
        let observed: Vec<_> = time
            .iter()
            .map(|&t| 100. / (1. + (-0.5 * (t - 8.)).exp()) + 2. + 0.1 * t.sin())
            .collect();
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time,"observed":observed},
                "constructor_kwargs":{"base_model":"logistic","smoothing":0.4}
            }),
        );
    }
    if id == "latent_process_reference" {
        let time = [0_f64, 1., 2., 3., 4., 5., 6.];
        let mut maximum = 0_f64;
        let observed: Vec<_> = time
            .iter()
            .zip([2., 2., -1., -4., 1., 6., 6.])
            .map(|(&t, offset)| {
                maximum = maximum.max(100. / (1. + (-0.5 * (t - 3.)).exp()) + offset);
                maximum
            })
            .collect();
        return fixed_prediction(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time},
                "constructor_kwargs":{"base_model":"logistic","latent_time":[1.,3.,5.],"latent_values":[2.,-4.,6.]},
                "parameters":{"base_L":100.,"base_k":0.5,"base_x0":3.,"latent_smoothing":0.3,"latent_noise":2.}
            }),
            json!(observed),
        );
    }
    if id == "policy_hazard_fit" {
        let time: Vec<f64> = (0..26).map(f64::from).collect();
        let observed: Vec<_> = time
            .iter()
            .map(|&t| {
                let decay = (-0.33 * t).exp();
                100. * (1. - decay) / (1. + 10. * decay)
            })
            .collect();
        return fit_and_forecast(
            id,
            model,
            family,
            json!({
                "inputs":{"time":time,"observed":observed},
                "constructor_kwargs":{"base_model":"bass","decay":0.3}
            }),
        );
    }
    if id == "policy_hazard_reference" {
        return policy_hazard_reference(id, model, family);
    }
    if id == "network_spillover_reference" {
        return network_spillover_reference(id, model, family);
    }
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

fn network_spillover_reference(id: &str, model: &str, family: &str) -> Result<Value, String> {
    let time = [0_f64, 1., 5., 20.];
    let bass = |t: f64, p: f64, q: f64, market: f64| {
        let decay = (-(p + q) * t).exp();
        market * (1. - decay) / (1. + (q / p) * decay)
    };
    let observed: Vec<_> = time
        .iter()
        .map(|&t| {
            let a = bass(t, 0.03, 0.3, 100.);
            let b = bass(t, 0.02, 0.2, 200.);
            let c = bass(t, 0.04, 0.4, 50.);
            [a + 0.2 * (2. * b + c) / 3., b + 0.2 * a, c]
        })
        .collect();
    fixed_prediction(
        id,
        model,
        family,
        json!({
            "inputs":{"time":time},
            "constructor_kwargs":{"base_model":"bass","adjacency":[[0.,2.,1.],[1.,0.,0.],[0.,0.,0.]],
                "node_labels":["A","B","isolated"]},
            "parameters":{"spillover_strength":0.2,"node_0_p":0.03,"node_0_q":0.3,"node_0_m":100.,
                "node_1_p":0.02,"node_1_q":0.2,"node_1_m":200.,
                "node_2_p":0.04,"node_2_q":0.4,"node_2_m":50.}
        }),
        json!(observed),
    )
}

fn policy_hazard_reference(id: &str, model: &str, family: &str) -> Result<Value, String> {
    let time = [0_f64, 2., 3., 4., 10.];
    let mut maximum = 0_f64;
    let observed: Vec<_> = time
        .iter()
        .map(|&t| {
            let effect: f64 = [(2., 0.5), (4., -0.2)]
                .into_iter()
                .filter(|(start, _)| t >= *start)
                .map(|(start, amplitude)| amplitude * (-0.5 * (t - start)).exp())
                .sum();
            maximum = maximum.max(100. / (1. + (-0.5 * (t - 3.)).exp()) * (1. + effect));
            maximum
        })
        .collect();
    fixed_prediction(
        id,
        model,
        family,
        json!({
            "inputs":{"time":time},
            "constructor_kwargs":{"base_model":"logistic","event_times":[2.,4.],"event_effects":[0.5,-0.2]},
            "parameters":{"decay":0.5,"base_L":100.,"base_k":0.5,"base_x0":3.}
        }),
        json!(observed),
    )
}

fn fixed_prediction(
    id: &str,
    model: &str,
    family: &str,
    payload: Value,
    observed: Value,
) -> Result<Value, String> {
    let time = payload["inputs"]["time"].clone();
    let prediction = crate::KernelBinding::new()
        .predict_model(model, payload)
        .map_err(|error| format!("benchmark prediction failed: {}", error.message))?
        .result
        .ok_or("benchmark prediction returned no result")?;
    Ok(
        json!({"case":case_descriptor(id,model,family),"time":time,"observed":observed,
        "fit":null,"prediction":prediction,"scope":"fixed_parameter_analytic_reference"}),
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
                .ok_or_else(|| format!("case_ids must contain 1 to {} case IDs", CASES.len()))?;
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
        "scope":"synthetic_benchmark_suite"}),
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
