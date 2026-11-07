use crate::{KERNEL_SCHEMA_VERSION, KernelBinding, batch_workers, run_ordered_chunks};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    model_key: String,
    simulation: Value,
    notes: Option<String>,
    specification: Option<ScenarioSpecification>,
}

#[derive(Deserialize, Serialize)]
struct ScenarioSpecification {
    name: String,
    description: String,
    time_horizon: f64,
    time_unit: String,
    reference_year: i64,
    market_size: f64,
    initial_adoption: f64,
    #[serde(flatten)]
    kind: ScenarioKind,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "scenario_type", rename_all = "snake_case", deny_unknown_fields)]
enum ScenarioKind {
    Baseline,
    Intervention {
        intervention_type: String,
        intervention_start_time: f64,
        intervention_magnitude: f64,
    },
    Substitution {
        incumbent_name: String,
        entrant_name: String,
        substitution_rate: f64,
    },
    Competition {
        num_competitors: usize,
        competitor_names: Vec<String>,
        market_share_initial: Vec<f64>,
    },
    Network {
        network_type: String,
        num_nodes: usize,
        average_degree: f64,
    },
}

impl ScenarioSpecification {
    fn validate(&self) -> Result<(), &'static str> {
        if self.time_horizon <= 0. || self.market_size <= 0. {
            return Err("time_horizon and market_size must be positive");
        }
        if !(0. ..=1.).contains(&self.initial_adoption) {
            return Err("initial_adoption must be in [0, 1]");
        }
        match &self.kind {
            ScenarioKind::Intervention {
                intervention_start_time,
                ..
            } if *intervention_start_time < 0. => {
                Err("intervention_start_time must be nonnegative")
            }
            ScenarioKind::Substitution {
                substitution_rate, ..
            } if !(0. ..=1.).contains(substitution_rate) => {
                Err("substitution_rate must be in [0, 1]")
            }
            ScenarioKind::Competition {
                num_competitors,
                competitor_names,
                market_share_initial,
            } if competitor_names.len() != *num_competitors
                || market_share_initial.len() != *num_competitors
                || market_share_initial
                    .iter()
                    .any(|share| !(0. ..=1.).contains(share))
                || (market_share_initial.iter().sum::<f64>() - 1.).abs() > 1e-6 =>
            {
                Err(
                    "competition names and shares must match num_competitors and shares must sum to one",
                )
            }
            ScenarioKind::Network {
                num_nodes,
                average_degree,
                ..
            } if *num_nodes == 0 || *average_degree <= 0. => {
                Err("num_nodes and average_degree must be positive")
            }
            _ => Ok(()),
        }
    }
}

pub(crate) fn execute_grid(request: &Value) -> Value {
    let envelope = |result: Value, error: Value| {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"execute_scenario_grid",
            "result":result,"error":error,"metadata":{"runtime":"rust_native"}})
    };
    let invalid = |message: &str| {
        envelope(
            Value::Null,
            json!({"code":"invalid_request","message":message}),
        )
    };
    if request["schema_version"] != KERNEL_SCHEMA_VERSION {
        return invalid("execute_scenario_grid requires schema 2.0");
    }
    let Some(workers) = batch_workers(&request["payload"]) else {
        return invalid("workers must be an integer in 1..=32");
    };
    let include_metadata = match request["payload"].get("include_execution_metadata") {
        None => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => return invalid("include_execution_metadata must be a boolean"),
    };
    let scenarios: Vec<Scenario> =
        match serde_json::from_value(request["payload"]["scenarios"].clone()) {
            Ok(scenarios) => scenarios,
            Err(error) => return invalid(&format!("invalid scenario grid: {error}")),
        };
    let mut ids = HashSet::new();
    if scenarios.iter().any(|scenario| {
        scenario.id.trim().is_empty()
            || scenario.model_key.trim().is_empty()
            || !ids.insert(&scenario.id)
    }) {
        return invalid(
            "scenario IDs must be nonblank and unique, and model keys must be nonblank",
        );
    }
    for scenario in &scenarios {
        if let Some(specification) = &scenario.specification
            && let Err(message) = specification.validate()
        {
            return invalid(&format!("scenario '{}': {message}", scenario.id));
        }
    }
    let comparisons: Option<Vec<ComparisonRequest>> = match request["payload"].get("comparisons") {
        None => None,
        Some(value) => match serde_json::from_value(value.clone()) {
            Ok(comparisons) => Some(comparisons),
            Err(error) => return invalid(&format!("invalid comparison list: {error}")),
        },
    };
    let binding = KernelBinding::new();
    let row = |scenario: &Scenario, result: Option<Value>, error: Value| {
        let mut row =
            json!({"id":scenario.id,"model_key":scenario.model_key,"result":result,"error":error});
        if let Some(specification) = &scenario.specification {
            row["specification"] = json!(specification);
        }
        if include_metadata {
            row["execution_metadata"] = json!({
                "kernel_version":env!("CARGO_PKG_VERSION"),
                "simulation":scenario.simulation,
                "notes":scenario.notes,
                "kernel":null,
                "started_at_unix_seconds":null,
                "elapsed_seconds":null,
            });
        }
        row
    };
    let executions = run_ordered_chunks(
        &scenarios,
        workers,
        |chunk| {
            chunk
                .iter()
                .map(|scenario| {
                    let timing = include_metadata.then(|| (SystemTime::now(), Instant::now()));
                    let execution =
                        binding.simulate_model(&scenario.model_key, scenario.simulation.clone());
                    let timing = timing.map(|(start, clock)| {
                        (
                            start,
                            clock.elapsed().as_secs_f64(),
                        )
                    });
                    let (mut row, kernel) = match execution {
                        Ok(response) => (
                            row(scenario, response.result, Value::Null),
                            response.metadata,
                        ),
                        Err(error) => (row(scenario, None, json!(error)), Value::Null),
                    };
                    if let Some((start, elapsed)) = timing {
                        let metadata = &mut row["execution_metadata"];
                        metadata["started_at_unix_seconds"] = json!(
                            start.duration_since(UNIX_EPOCH).ok().map(|duration| duration.as_secs_f64())
                        );
                        metadata["elapsed_seconds"] = json!(elapsed);
                        metadata["kernel"] = kernel;
                        if row["error"].is_null() {
                            let timestamp = chrono::DateTime::<chrono::Utc>::from(start).to_rfc3339();
                            row["artifact"] = json!({
                                "scenario":scenario.specification,
                                "seed":row["result"]["metadata"]["seed"],
                                "model_type":scenario.model_key,
                                "version":env!("CARGO_PKG_VERSION"),
                                "notes":scenario.notes,
                                "timestamp":timestamp,
                                "replay_request":{"schema_version":KERNEL_SCHEMA_VERSION,
                                    "operation":"execute_scenario_grid","payload":{"scenarios":[{
                                        "id":scenario.id,"model_key":scenario.model_key,
                                        "simulation":scenario.simulation,"specification":scenario.specification,
                                        "notes":scenario.notes
                                    }]}},
                            });
                        }
                    }
                    row
                })
                .collect()
        },
        |scenario| {
            row(
                scenario,
                None,
                json!({"code":"batch_worker_panicked","message":"Native scenario worker failed"}),
            )
        },
    );
    let mut result = json!({"executions":executions});
    if let Some(comparisons) = comparisons {
        let paths: HashMap<_, _> = scenarios
            .iter()
            .zip(&executions)
            .map(|(scenario, row)| (scenario.id.as_str(), (scenario, &row["result"])))
            .collect();
        let rows: Vec<_> = comparisons
            .iter()
            .map(|comparison| {
                let (mut row, outcome) = match comparison {
                    ComparisonRequest::Pair(pair) => (
                        json!({"baseline":pair.baseline,"alternative":pair.alternative}),
                        compare_paths(&paths, pair),
                    ),
                    ComparisonRequest::Ensemble(ensemble) => (
                        json!({"pairs":ensemble.pairs}),
                        compare_ensemble(&paths, ensemble),
                    ),
                };
                let (result, error) = match outcome {
                    Ok(result) => (result, Value::Null),
                    Err(message) => (
                        Value::Null,
                        json!({"code":"invalid_comparison","message":message}),
                    ),
                };
                row["result"] = result;
                row["error"] = error;
                row
            })
            .collect();
        result["comparisons"] = json!(rows);
    }
    envelope(result, Value::Null)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Comparison {
    baseline: String,
    alternative: String,
    threshold: Option<f64>,
    relative_margin: Option<f64>,
}

type ScenarioPaths<'a> = HashMap<&'a str, (&'a Scenario, &'a Value)>;

#[derive(Deserialize)]
#[serde(untagged)]
enum ComparisonRequest {
    Pair(Comparison),
    Ensemble(EnsembleComparison),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EnsembleComparison {
    pairs: Vec<ReplicatePair>,
    interval_mass: f64,
    bootstrap: Option<EnsembleBootstrap>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EnsembleBootstrap {
    draws: usize,
    seed: u64,
}

#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct ReplicatePair {
    baseline: String,
    alternative: String,
}

fn common_time_basis<'a>(
    paths: &ScenarioPaths<'a>,
    ids: impl IntoIterator<Item = &'a str>,
) -> Result<Option<(&'a str, i64)>, &'static str> {
    let mut common = None;
    for id in ids {
        let (scenario, _) = paths.get(id).ok_or("unknown scenario ID")?;
        let basis = scenario
            .specification
            .as_ref()
            .map(|spec| (spec.time_unit.as_str(), spec.reference_year));
        if common.is_some_and(|expected| expected != basis) {
            return Err(
                "scenario comparisons require matching time units and reference years when specified",
            );
        }
        common = Some(basis);
    }
    Ok(common.flatten())
}

fn add_time_basis(result: &mut Value, basis: Option<(&str, i64)>) {
    if let Some((unit, year)) = basis {
        result["time_basis"] = json!({"time_unit":unit,"reference_year":year});
    }
}

fn scalar_path(paths: &ScenarioPaths<'_>, id: &str) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    let (scenario, result) = paths.get(id).ok_or("unknown scenario ID")?;
    let inputs = &scenario.simulation["inputs"];
    let time = inputs.get("time").unwrap_or(&inputs["t"]);
    let time: Vec<f64> = serde_json::from_value(time.clone())
        .map_err(|_| "comparison requires a numeric time grid")?;
    let values: Vec<f64> = serde_json::from_value(result["values"].clone())
        .map_err(|_| "comparison requires a successful scalar simulation")?;
    if time.is_empty()
        || time.len() != values.len()
        || time.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("comparison requires one value per strictly increasing sample time");
    }
    Ok((time, values))
}

fn compare_paths(
    paths: &ScenarioPaths<'_>,
    comparison: &Comparison,
) -> Result<Value, &'static str> {
    let basis = common_time_basis(
        paths,
        [
            comparison.baseline.as_str(),
            comparison.alternative.as_str(),
        ],
    )?;
    let (time, baseline) = scalar_path(paths, &comparison.baseline)?;
    let (alternative_time, alternative) = scalar_path(paths, &comparison.alternative)?;
    if time != alternative_time {
        return Err("scenario comparisons require identical time grids");
    }
    let (difference, percentage_change) = crate::causal::compare_values(&baseline, &alternative)?;
    let mut result =
        json!({"time":time,"difference":difference,"percentage_change":percentage_change});
    add_time_basis(&mut result, basis);
    let last = time.len() - 1;
    let (baseline_final, alternative_final) = (baseline[last], alternative[last]);
    result["ranking"] = json!({"metric":"final_adoption",
        "baseline_rank":baseline_final,"alternative_rank":alternative_final});
    result["incremental_effect"] = json!({"baseline_value":baseline_final,
        "alternative_value":alternative_final,"absolute_increase":difference[last],
        "relative_increase_percent":percentage_change[last]});
    if let Some(margin) = comparison.relative_margin {
        if !(0. ..=1.).contains(&margin) {
            return Err("relative_margin must be in [0, 1]");
        }
        let bounds = |value: f64| {
            let delta = value * margin;
            let upper = value + delta;
            if value < 0. || !upper.is_finite() {
                return Err("relative margin bounds require nonnegative values and finite bounds");
            }
            Ok([(value - delta).max(0.), upper])
        };
        result["relative_margin_bounds"] = json!({"method":"fixed_relative_margin",
            "relative_margin":margin,"baseline":bounds(baseline_final)?,
            "alternative":bounds(alternative_final)?});
    }
    if let Some(threshold) = comparison.threshold {
        let first_sample = |values: &[f64]| {
            time.iter()
                .zip(values)
                .find_map(|(&t, &value)| (value >= threshold).then_some(t))
        };
        result["threshold_timing"] =
            json!({"baseline":first_sample(&baseline),"alternative":first_sample(&alternative)});
    }
    Ok(result)
}

fn compare_ensemble(
    paths: &ScenarioPaths<'_>,
    comparison: &EnsembleComparison,
) -> Result<Value, &'static str> {
    let mass = comparison.interval_mass;
    if mass <= 0. || mass >= 1. || comparison.pairs.len() < 2 {
        return Err("ensemble requires at least two pairs and interval_mass in (0, 1)");
    }
    let basis = common_time_basis(
        paths,
        comparison
            .pairs
            .iter()
            .flat_map(|pair| [pair.baseline.as_str(), pair.alternative.as_str()]),
    )?;
    let mut used = HashSet::new();
    let mut grid = None;
    let mut differences = Vec::with_capacity(comparison.pairs.len());
    for pair in &comparison.pairs {
        if !used.insert(&pair.baseline) || !used.insert(&pair.alternative) {
            return Err("each scenario ID can appear only once per ensemble");
        }
        let (time, baseline) = scalar_path(paths, &pair.baseline)?;
        let (alternative_time, alternative) = scalar_path(paths, &pair.alternative)?;
        if time != alternative_time || grid.as_ref().is_some_and(|expected| expected != &time) {
            return Err("ensemble comparisons require identical time grids across all replicates");
        }
        let last = time.len() - 1;
        grid.get_or_insert(time);
        let difference = alternative[last] - baseline[last];
        if !difference.is_finite() {
            return Err("ensemble difference must be finite");
        }
        differences.push(difference);
    }
    // Scale before squaring so finite effects near the floating-point limits
    // can still yield a representable standard error.
    let scale = differences.iter().map(|x| x.abs()).fold(0., f64::max);
    let (mean, standard_error) = if scale == 0. {
        (0., 0.)
    } else {
        let n = differences.len() as f64;
        let mean = differences.iter().map(|x| x / scale).sum::<f64>() / n;
        let centered_squares = differences
            .iter()
            .map(|x| (x / scale - mean).powi(2))
            .sum::<f64>();
        (
            mean * scale,
            (centered_squares / (n * (n - 1.))).sqrt() * scale,
        )
    };
    let mut sorted = differences.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let tail = (1. - mass) / 2.;
    let mut result = json!({"method":"paired_empirical_final_difference","quantile_method":"linear",
        "pair_count":differences.len(),"differences":differences,"interval_mass":mass,
        "mean_difference":mean,"mean_standard_error":standard_error,
        "standard_error_assumption":"independent replicate pairs",
        "median_difference":linear_quantile(&sorted, 0.5),"empirical_interval":[linear_quantile(&sorted, tail),linear_quantile(&sorted, 1.-tail)]});
    if let Some(options) = &comparison.bootstrap {
        result["bootstrap"] = bootstrap_mean(&differences, options, scale, mass)?;
    }
    add_time_basis(&mut result, basis);
    Ok(result)
}

fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let rank = probability * (sorted.len() - 1) as f64;
    sorted[rank.floor() as usize] * (1. - rank.fract())
        + sorted[rank.ceil() as usize] * rank.fract()
}

fn bootstrap_mean(
    differences: &[f64],
    options: &EnsembleBootstrap,
    scale: f64,
    mass: f64,
) -> Result<Value, &'static str> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};

    if !(2..=10_000).contains(&options.draws) {
        return Err("bootstrap draws must be between 2 and 10000");
    }
    if differences.len() > 10_000_000 / options.draws {
        return Err("bootstrap workload exceeds 10000000 sampled pairs");
    }
    let mut rng = DeterministicStream::from_run_seed(options.seed);
    let index = Uniform::new(0, differences.len()).map_err(|_| "empty paired sample")?;
    let divisor = if scale == 0. { 1. } else { scale };
    let scaled: Vec<_> = differences.iter().map(|value| value / divisor).collect();
    let mut means: Vec<_> = (0..options.draws)
        .map(|_| {
            let sum: f64 = (0..scaled.len()).map(|_| scaled[rng.sample(index)]).sum();
            (sum / scaled.len() as f64) * divisor
        })
        .collect();
    means.sort_unstable_by(f64::total_cmp);
    let tail = (1. - mass) / 2.;
    Ok(
        json!({"method":"paired_percentile","seed":options.seed,"draws":options.draws,
        "interval_mass":mass,"assumption":"independent replicate pairs",
        "mean_interval":[linear_quantile(&means, tail),linear_quantile(&means, 1.-tail)]}),
    )
}

const SCENARIO_FAMILIES: [&str; 5] = [
    "baseline",
    "intervention",
    "substitution",
    "competition",
    "network",
];

pub(crate) fn discover(request: &Value) -> Value {
    let envelope = |result: Value, error: Value| {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"discover_scenarios",
            "result":result,"error":error,"metadata":{"runtime":"rust_native"}})
    };
    let invalid = |message: &str| {
        envelope(
            Value::Null,
            json!({"code":"invalid_request","message":message}),
        )
    };
    if request["schema_version"] != KERNEL_SCHEMA_VERSION {
        return invalid("discover_scenarios requires schema 2.0");
    }
    let key = match request["payload"].get("key") {
        None => None,
        Some(Value::String(key)) if !key.trim().is_empty() => Some(key.as_str()),
        Some(_) => return invalid("scenario key must be a nonblank string"),
    };
    if let Some(key) = key
        && !SCENARIO_FAMILIES.contains(&key)
    {
        return envelope(
            Value::Null,
            json!({"code":"unknown_scenario",
            "message":format!("unknown scenario family '{key}'"),"available":SCENARIO_FAMILIES}),
        );
    }
    let scenarios: Vec<_> = SCENARIO_FAMILIES
        .into_iter()
        .filter(|family| key.is_none_or(|key| key == *family))
        .map(|family| {
            json!({"key":family,"scenario_type":family,"stability":"experimental",
                "operation":"execute_scenario_grid","execution_input":"explicit_simulation_payload",
                "transport":"json","bindings":["rust","python","julia","r"]})
        })
        .collect();
    envelope(json!({"scenarios":scenarios}), Value::Null)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LegacyArtifact {
    scenario: ScenarioSpecification,
    seed: i128,
    model_type: String,
    version: String,
    notes: Option<String>,
    timestamp: String,
}

pub(crate) fn import_artifact(request: &Value) -> Value {
    let envelope = |result: Value, error: Value| {
        json!({"schema_version":KERNEL_SCHEMA_VERSION,"operation":"import_scenario_artifact",
            "result":result,"error":error,"metadata":{"runtime":"rust_native"}})
    };
    let invalid = |message: &str| {
        envelope(
            Value::Null,
            json!({"code":"invalid_request","message":message}),
        )
    };
    if request["schema_version"] != KERNEL_SCHEMA_VERSION {
        return invalid("import_scenario_artifact requires schema 2.0");
    }
    let artifact: LegacyArtifact =
        match serde_json::from_value(request["payload"]["artifact"].clone()) {
            Ok(artifact) => artifact,
            Err(error) => return invalid(&format!("invalid legacy artifact: {error}")),
        };
    if let Err(message) = artifact.scenario.validate() {
        return invalid(message);
    }
    if artifact.model_type.trim().is_empty() || artifact.version.trim().is_empty() {
        return invalid("legacy artifact requires nonblank model_type and version");
    }
    if chrono::DateTime::parse_from_rfc3339(&artifact.timestamp).is_err() {
        return invalid("legacy artifact timestamp must be RFC 3339 with a time offset");
    }
    envelope(
        json!({"artifact":artifact,"format":"legacy_scenario_envelope",
        "replay_available":false,"replay_reason":"legacy artifact has no simulation payload"}),
        Value::Null,
    )
}
