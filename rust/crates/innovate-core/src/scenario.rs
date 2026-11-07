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

fn apply_specification(scenario: &mut Scenario) -> Result<(), &'static str> {
    let spec = scenario
        .specification
        .as_ref()
        .ok_or("scenario specification is required")?;
    if let ScenarioKind::Substitution {
        substitution_rate, ..
    } = &spec.kind
    {
        if scenario.model_key != "fixed_pool_replacement" {
            return Err("substitution translation requires fixed_pool_replacement");
        }
        let simulation = scenario
            .simulation
            .as_object_mut()
            .ok_or("simulation must be an object")?;
        let parameters = json!({"initial_incumbents":spec.market_size * spec.initial_adoption,
            "replacement_hazard":substitution_rate});
        if simulation
            .get("parameters")
            .is_some_and(|value| value != &parameters)
        {
            return Err("replacement parameters conflict with the scenario specification");
        }
        simulation.insert("parameters".into(), parameters);
        return Ok(());
    }
    if let ScenarioKind::Competition {
        num_competitors,
        market_share_initial,
        ..
    } = &spec.kind
    {
        return apply_competition_specification(
            &mut scenario.simulation,
            &scenario.model_key,
            *num_competitors,
            market_share_initial,
            spec.market_size * spec.initial_adoption,
        );
    }
    if scenario.model_key != "bass"
        || !matches!(
            spec.kind,
            ScenarioKind::Baseline
                | ScenarioKind::Intervention { .. }
                | ScenarioKind::Network { .. }
        )
    {
        return Err(
            "specification translation currently supports baseline, intervention and network Bass scenarios",
        );
    }
    let simulation = scenario
        .simulation
        .as_object_mut()
        .ok_or("simulation must be an object")?;
    let options = simulation
        .get_mut("simulation_options")
        .and_then(Value::as_object_mut)
        .ok_or("specification translation requires Kairos simulation_options")?;
    if let ScenarioKind::Network {
        network_type,
        num_nodes,
        average_degree,
    } = &spec.kind
    {
        apply_network_specification(
            options,
            network_type,
            *num_nodes,
            *average_degree,
            spec.initial_adoption,
        )?;
    } else {
        if !matches!(
            options.get("engine").and_then(Value::as_str),
            Some("kairos_abm" | "kairos_des")
        ) {
            return Err("baseline and intervention translation requires kairos_abm or kairos_des");
        }
        let population = options
            .get("population")
            .and_then(Value::as_u64)
            .filter(|n| (1..=100_000).contains(n))
            .ok_or("population must be in 1..=100000")?;
        let initial = (spec.initial_adoption * population as f64).round() as u64;
        if options
            .get("initial_adopters")
            .is_some_and(|value| value.as_u64() != Some(initial))
        {
            return Err("initial_adopters conflicts with the scenario specification");
        }
        options.insert("initial_adopters".into(), json!(initial));
    }
    if let ScenarioKind::Intervention {
        intervention_type,
        intervention_start_time,
        intervention_magnitude,
    } = &spec.kind
    {
        if *intervention_start_time > spec.time_horizon {
            return Err("intervention must start within the scenario time horizon");
        }
        let (delta_p, delta_q) = match intervention_type.as_str() {
            "innovation_delta" => (*intervention_magnitude, 0.),
            "imitation_delta" => (0., *intervention_magnitude),
            _ => return Err("Bass intervention type must be innovation_delta or imitation_delta"),
        };
        if options.contains_key("events") {
            return Err("explicit events conflict with scenario intervention translation");
        }
        options.insert(
            "events".into(),
            json!([{
                "time":intervention_start_time,"delta_p":delta_p,"delta_q":delta_q
            }]),
        );
    }

    let parameters = simulation
        .get_mut("parameters")
        .and_then(Value::as_object_mut)
        .ok_or("Bass scenario translation requires explicit parameters")?;
    if parameters
        .get("m")
        .is_some_and(|value| value.as_f64() != Some(spec.market_size))
    {
        return Err("Bass market size conflicts with the scenario specification");
    }
    parameters.insert("m".into(), json!(spec.market_size));
    Ok(())
}

fn apply_competition_specification(
    simulation: &mut Value,
    model: &str,
    competitors: usize,
    shares: &[f64],
    initial_market: f64,
) -> Result<(), &'static str> {
    if model != "multi_product" || !(1..=32).contains(&competitors) {
        return Err("competition translation requires multi_product with 1..=32 competitors");
    }
    let simulation = simulation
        .as_object_mut()
        .ok_or("simulation must be an object")?;
    if simulation.contains_key("state") {
        return Err("competition translation requires explicit parameters instead of fitted state");
    }
    let constructor = simulation
        .entry("constructor_kwargs")
        .or_insert_with(|| json!({}));
    let constructor = constructor
        .as_object_mut()
        .ok_or("constructor_kwargs must be an object")?;
    if constructor
        .get("n_products")
        .is_some_and(|n| n.as_u64() != Some(competitors as u64))
    {
        return Err("n_products conflicts with the scenario specification");
    }
    let initial: Vec<_> = shares.iter().map(|share| initial_market * share).collect();
    if let Some(values) = constructor.get("initial_values")
        && !values.as_array().is_some_and(|values| {
            values.len() == initial.len()
                && values
                    .iter()
                    .zip(&initial)
                    .all(|(given, expected)| given.as_f64() == Some(*expected))
        })
    {
        return Err("initial_values conflict with the scenario specification");
    }
    constructor.insert("n_products".into(), json!(competitors));
    constructor.insert("initial_values".into(), json!(initial));
    Ok(())
}

fn apply_network_specification(
    options: &mut serde_json::Map<String, Value>,
    network_type: &str,
    nodes: usize,
    average_degree: f64,
    initial_adoption: f64,
) -> Result<(), &'static str> {
    if !matches!(network_type, "custom" | "ring" | "random_fixed_edges")
        || options.get("engine").and_then(Value::as_str) != Some("kairos_network")
    {
        return Err("network translation requires a supported topology and kairos_network");
    }
    if nodes > 1000
        || options
            .get("population")
            .is_some_and(|value| value.as_u64() != Some(nodes as u64))
    {
        return Err("network population must match num_nodes and be in 1..=1000");
    }
    if network_type == "ring" {
        if nodes < 3
            || average_degree != 2.
            || options.contains_key("edges")
            || options.contains_key("edge_weights")
            || options
                .get("directed")
                .is_some_and(|value| value.as_bool() != Some(false))
        {
            return Err(
                "ring topology requires at least three nodes, degree two, and no explicit edges, weights or directed mode",
            );
        }
        let edges: Vec<_> = (0..nodes).map(|node| [node, (node + 1) % nodes]).collect();
        options.insert("edges".into(), json!(edges));
    }
    if network_type == "random_fixed_edges" {
        if options.contains_key("edges")
            || options.contains_key("edge_weights")
            || options
                .get("directed")
                .is_some_and(|value| value.as_bool() != Some(false))
        {
            return Err(
                "random_fixed_edges requires undirected mode without explicit edges or weights",
            );
        }
        let seed = options
            .remove("topology_seed")
            .and_then(|value| value.as_u64())
            .ok_or("random_fixed_edges requires an unsigned topology_seed")?;
        let edge_count = (average_degree * nodes as f64 / 2.).round() as usize;
        let max_edges = nodes * (nodes - 1) / 2;
        if edge_count == 0
            || edge_count > max_edges
            || (2. * edge_count as f64 / nodes as f64 - average_degree).abs() > 1e-9
        {
            return Err("average_degree must specify a feasible whole number of undirected edges");
        }
        options.insert("edges".into(), json!(random_edges(nodes, edge_count, seed)));
    }
    let edges = options
        .get("edges")
        .and_then(Value::as_array)
        .ok_or("custom topology requires explicit edges")?;
    let degree_factor = if options.get("directed").and_then(Value::as_bool) == Some(true) {
        1.
    } else {
        2.
    };
    let degree = degree_factor * edges.len() as f64 / nodes as f64;
    if (degree - average_degree).abs() > 1e-9 {
        return Err("average_degree does not match the initial explicit graph");
    }
    let initial = (initial_adoption * nodes as f64).round() as usize;
    let supplied = options
        .get("initial_agents")
        .map_or(Some(0), |value| value.as_array().map(Vec::len));
    if supplied != Some(initial) {
        return Err("explicit initial_agents must match the scenario's rounded adoption count");
    }
    options.insert("population".into(), json!(nodes));
    Ok(())
}

fn random_edges(nodes: usize, count: usize, seed: u64) -> Vec<[usize; 2]> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};

    let mut candidates: Vec<_> = (0..nodes)
        .flat_map(|a| ((a + 1)..nodes).map(move |b| [a, b]))
        .collect();
    let mut rng = DeterministicStream::from_run_seed(seed);
    for i in 0..count {
        let range = Uniform::new(i, candidates.len()).expect("a candidate remains");
        let index = rng.sample(range);
        candidates.swap(i, index);
    }
    candidates.truncate(count);
    candidates.sort_unstable();
    candidates
}

fn simulate_scenario(
    binding: &KernelBinding,
    scenario: &Scenario,
) -> Result<crate::KernelResponse, crate::KernelBindingError> {
    if scenario.model_key != "fixed_pool_replacement" {
        return binding.simulate_model(&scenario.model_key, scenario.simulation.clone());
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReplacementInput {
        inputs: ReplacementTimes,
        parameters: ReplacementParameters,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReplacementTimes {
        #[serde(alias = "t")]
        time: Vec<f64>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReplacementParameters {
        initial_incumbents: f64,
        replacement_hazard: f64,
    }
    let op = crate::KernelOperation::SimulateModel;
    let invalid = |message| crate::KernelBindingError::invalid_request(op, message);
    let input = ReplacementInput::deserialize(&scenario.simulation)
        .map_err(|error| invalid(format!("invalid replacement payload: {error}")))?;
    let time = input.inputs.time;
    let pool = input.parameters.initial_incumbents;
    let hazard = input.parameters.replacement_hazard;
    if pool < 0.
        || !(0. ..=1.).contains(&hazard)
        || time.is_empty()
        || time[0] < 0.
        || time.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(invalid("replacement requires a nonnegative pool, hazard in [0, 1] and increasing nonnegative times".into()));
    }
    let rows: Vec<_> = time
        .iter()
        .map(|t| {
            let exponent = -hazard * t;
            [pool * exponent.exp(), pool * -exponent.exp_m1()]
        })
        .collect();
    Ok(crate::KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: Some(scenario.model_key.clone()),
        result: Some(
            json!({"rows":rows,"time":time,"columns":["incumbent","entrant"],
            "metadata":{"replacement_semantics":"constant one-way hazard per scenario time unit; fixed initial adopter pool"}}),
        ),
        error: None,
        metadata: json!({"runtime":"rust_native","engine":"analytic_replacement"}),
    })
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
    let apply_specifications = match request["payload"].get("apply_specifications") {
        None => false,
        Some(Value::Bool(enabled)) => *enabled,
        Some(_) => return invalid("apply_specifications must be a boolean"),
    };
    let mut scenarios: Vec<Scenario> =
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
    if apply_specifications {
        for scenario in &mut scenarios {
            if let Err(message) = apply_specification(scenario) {
                return invalid(&format!("scenario '{}': {message}", scenario.id));
            }
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
                    let execution = simulate_scenario(&binding, scenario);
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
            let mut capability = json!({"key":family,"scenario_type":family,"stability":"experimental",
                "operation":"execute_scenario_grid","execution_input":"explicit_simulation_payload",
                "transport":"json","bindings":["rust","python","julia","r"]});
            if matches!(family, "baseline" | "intervention") {
                capability["specification_translation"] = json!({
                    "option":"apply_specifications","model_key":"bass",
                    "engines":["kairos_abm","kairos_des"]
                });
                if family == "intervention" {
                    capability["specification_translation"]["intervention_types"] =
                        json!(["innovation_delta","imitation_delta"]);
                }
            }
            if family == "network" {
                capability["specification_translation"] = json!({
                    "option":"apply_specifications","model_key":"bass",
                    "engines":["kairos_network"],"network_types":["custom","ring","random_fixed_edges"]
                });
            }
            if family == "competition" {
                capability["specification_translation"] = json!({
                    "option":"apply_specifications","model_key":"multi_product"
                });
            }
            if family == "substitution" {
                capability["specification_translation"] = json!({
                    "option":"apply_specifications","model_key":"fixed_pool_replacement",
                    "rate_semantics":"constant one-way replacement hazard per scenario time unit"
                });
            }
            capability
        })
        .collect();
    envelope(json!({"scenarios":scenarios}), Value::Null)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportedArtifact {
    scenario: Option<ScenarioSpecification>,
    #[serde(rename = "seed")]
    seed: Option<i128>,
    model_type: String,
    version: String,
    #[serde(rename = "notes")]
    _notes: Option<String>,
    timestamp: String,
    replay_request: Option<Value>,
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
    let original = &request["payload"]["artifact"];
    let artifact = match ImportedArtifact::deserialize(original) {
        Ok(artifact) => artifact,
        Err(error) => return invalid(&format!("invalid scenario artifact: {error}")),
    };
    if let Some(specification) = &artifact.scenario
        && let Err(message) = specification.validate()
    {
        return invalid(message);
    }
    if artifact.model_type.trim().is_empty() || artifact.version.trim().is_empty() {
        return invalid("scenario artifact requires nonblank model_type and version");
    }
    if chrono::DateTime::parse_from_rfc3339(&artifact.timestamp).is_err() {
        return invalid("scenario artifact timestamp must be RFC 3339 with a time offset");
    }
    let (format, reason) = if let Some(replay) = &artifact.replay_request {
        let reason = match validate_replay(original, &artifact, replay) {
            Ok(reason) => reason,
            Err(error) => return invalid(error),
        };
        ("native_scenario_envelope", reason)
    } else {
        if artifact.scenario.is_none() || artifact.seed.is_none() {
            return invalid("legacy artifact requires scenario and seed");
        }
        (
            "legacy_scenario_envelope",
            Some("legacy artifact has no simulation payload"),
        )
    };
    envelope(
        json!({"artifact":original,"format":format,
        "replay_available":reason.is_none(),"replay_reason":reason}),
        Value::Null,
    )
}

fn validate_replay(
    original: &Value,
    artifact: &ImportedArtifact,
    replay: &Value,
) -> Result<Option<&'static str>, &'static str> {
    if replay["operation"] != "execute_scenario_grid" {
        return Err("artifact replay must use execute_scenario_grid");
    }
    let schema = replay["schema_version"]
        .as_str()
        .filter(|version| !version.is_empty())
        .ok_or("artifact replay requires a schema version")?;
    if schema != KERNEL_SCHEMA_VERSION {
        return Ok(Some("artifact replay uses an unsupported kernel schema"));
    }
    let payload = replay["payload"]
        .as_object()
        .filter(|payload| payload.len() == 1 && payload.contains_key("scenarios"))
        .ok_or("artifact replay payload must contain only scenarios")?;
    let entries = payload["scenarios"]
        .as_array()
        .filter(|entries| entries.len() == 1)
        .ok_or("artifact replay must contain exactly one scenario")?;
    let entry =
        Scenario::deserialize(&entries[0]).map_err(|_| "invalid artifact replay scenario")?;
    if entry.id.trim().is_empty()
        || !entry.simulation.is_object()
        || entry.model_key != artifact.model_type
        || entries[0]["specification"] != original["scenario"]
        || entries[0]["notes"] != original["notes"]
    {
        return Err("artifact replay scenario does not match the envelope");
    }
    let seed_matches = match entry.simulation["simulation_options"].get("seed") {
        Some(seed) => seed.is_u64() && seed == &original["seed"],
        None => artifact.seed.is_none(),
    };
    if !seed_matches {
        return Err("artifact replay seed does not match the envelope");
    }
    Ok((artifact.version != env!("CARGO_PKG_VERSION"))
        .then_some("artifact was created by a different kernel version"))
}
