use crate::numerics::linear_quantile;
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
    #[serde(skip)]
    submitted_simulation: Option<Value>,
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
        if self.time_unit.trim().is_empty() {
            return Err("time_unit must be nonblank");
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
                network_type,
            } if *num_nodes == 0
                || *average_degree < 0.
                || (*average_degree == 0. && network_type != "custom") =>
            {
                Err(
                    "num_nodes must be positive and average_degree must be positive except for custom graphs",
                )
            }
            ScenarioKind::Network {
                num_nodes,
                average_degree,
                ..
            } if *average_degree > num_nodes.saturating_sub(1) as f64 => {
                Err("average_degree cannot exceed num_nodes minus one")
            }
            _ => Ok(()),
        }
    }
}

fn validate_network_horizon(scenario: &Scenario) -> Result<(), &'static str> {
    let Some(spec) = &scenario.specification else {
        return Ok(());
    };
    if matches!(spec.kind, ScenarioKind::Network { .. })
        && scenario.simulation.get("inputs").is_some_and(|inputs| {
            inputs
                .get("time")
                .or_else(|| inputs.get("t"))
                .and_then(Value::as_array)
                .is_some_and(|time| {
                    time.iter()
                        .filter_map(Value::as_f64)
                        .any(|sample| sample > spec.time_horizon)
                })
        })
    {
        return Err("network observation time exceeds the declared scenario horizon");
    }
    Ok(())
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
    if !matches!(
        network_type,
        "custom" | "ring" | "random_fixed_edges" | "scale_free"
    ) || options.get("engine").and_then(Value::as_str) != Some("kairos_network")
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
    if matches!(network_type, "random_fixed_edges" | "scale_free") {
        if options.contains_key("edges")
            || options.contains_key("edge_weights")
            || options
                .get("directed")
                .is_some_and(|value| value.as_bool() != Some(false))
        {
            return Err(
                "seeded topology requires undirected mode without explicit edges or weights",
            );
        }
        let seed = options
            .remove("topology_seed")
            .and_then(|value| value.as_u64())
            .ok_or("seeded topology requires an unsigned topology_seed")?;
        let edge_count = (average_degree * nodes as f64 / 2.).round() as usize;
        let max_edges = nodes * (nodes - 1) / 2;
        if edge_count == 0
            || edge_count > max_edges
            || (2. * edge_count as f64 / nodes as f64 - average_degree).abs() > 1e-9
        {
            return Err("average_degree must specify a feasible whole number of undirected edges");
        }
        let edges = if network_type == "scale_free" {
            let links = (1..nodes.min(33))
                .find(|&links| links * nodes - links * (links + 1) / 2 == edge_count)
                .ok_or("scale_free mean degree must match a clique-seeded graph with 1..=32 links per new node")?;
            preferential_edges(nodes, links, seed)
        } else {
            random_edges(nodes, edge_count, seed)
        };
        options.insert("edges".into(), json!(edges));
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

fn preferential_edges(nodes: usize, links: usize, seed: u64) -> Vec<[usize; 2]> {
    use kairo_ecs_rng::{DeterministicStream, rand_distr::Uniform};

    let clique = links + 1;
    let mut edges: Vec<_> = (0..clique)
        .flat_map(|a| ((a + 1)..clique).map(move |b| [a, b]))
        .collect();
    let mut degrees = vec![links; clique];
    let mut rng = DeterministicStream::from_run_seed(seed);
    for node in clique..nodes {
        let mut chosen = Vec::with_capacity(links);
        let mut total: usize = degrees.iter().sum();
        for _ in 0..links {
            let mut draw = rng.sample(Uniform::new(0, total).expect("positive available degree"));
            let target = degrees
                .iter()
                .position(|&degree| {
                    if draw < degree {
                        true
                    } else {
                        draw -= degree;
                        false
                    }
                })
                .expect("draw is within available degree");
            let degree = std::mem::take(&mut degrees[target]);
            total -= degree;
            chosen.push((target, degree));
            edges.push([target, node]);
        }
        for (target, degree) in chosen {
            degrees[target] = degree + 1;
        }
        degrees.push(links);
    }
    edges.sort_unstable();
    edges
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
        if let Err(message) = validate_network_horizon(scenario) {
            return invalid(&format!("scenario '{}': {message}", scenario.id));
        }
    }
    if apply_specifications {
        for scenario in &mut scenarios {
            if include_metadata {
                scenario.submitted_simulation = Some(scenario.simulation.clone());
            }
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
    let ranking: Option<Vec<String>> = match request["payload"].get("rank_scenarios") {
        None => None,
        Some(value) => match serde_json::from_value(value.clone()) {
            Ok(ids) => Some(ids),
            Err(_) => return invalid("rank_scenarios must contain scenario IDs"),
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
            if let Some(submitted) = &scenario.submitted_simulation {
                row["execution_metadata"]["submitted_simulation"] = submitted.clone();
            }
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
                            let seed = row["result"]["metadata"].get("seed").unwrap_or_else(|| {
                                if row["result"]["status"] == "paused" {
                                    &scenario.simulation["simulation_options"]["seed"]
                                } else {
                                    &Value::Null
                                }
                            });
                            row["artifact"] = json!({
                                "scenario":scenario.specification,
                                "seed":seed,
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
                            if let Some(submitted) = &scenario.submitted_simulation {
                                row["artifact"]["submitted_simulation"] = submitted.clone();
                            }
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
    if comparisons.is_some() || ranking.is_some() {
        let paths: HashMap<_, _> = scenarios
            .iter()
            .zip(&executions)
            .map(|(scenario, row)| (scenario.id.as_str(), (scenario, &row["result"])))
            .collect();
        if let Some(ids) = ranking {
            let metric = request["payload"]
                .get("rank_metric")
                .map_or(Some("final_adoption"), Value::as_str);
            result["ranking"] =
                match metric
                    .ok_or("rank_metric must be a string")
                    .and_then(|metric| {
                        rank_paths(&paths, &ids, metric, &request["payload"]["rank_threshold"])
                    }) {
                    Ok(ranking) => ranking,
                    Err(message) => {
                        json!({"rows":null,"error":{"code":"invalid_ranking","message":message}})
                    }
                };
        }
        if let Some(comparisons) = comparisons {
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
    }
    envelope(result, Value::Null)
}

fn rank_paths(
    paths: &ScenarioPaths<'_>,
    ids: &[String],
    metric: &str,
    threshold: &Value,
) -> Result<Value, &'static str> {
    if ids.len() < 2 || ids.iter().collect::<std::collections::HashSet<_>>().len() != ids.len() {
        return Err("ranking requires at least two distinct scenario IDs");
    }
    let basis = common_time_basis(paths, ids.iter().map(String::as_str))?;
    let threshold = if metric == "time_to_threshold" {
        Some(
            threshold
                .as_f64()
                .ok_or("time_to_threshold requires numeric rank_threshold")?,
        )
    } else {
        None
    };
    let mut grid = None;
    let mut scores = Vec::with_capacity(ids.len());
    for id in ids {
        let (time, values) = scalar_path(paths, id)?;
        if grid.as_ref().is_some_and(|expected| expected != &time) {
            return Err("scenario ranking requires identical time grids");
        }
        let score = match metric {
            "final_adoption" => values.last().copied(),
            "adoption_area" => Some(adoption_area(&time, &values)?),
            "time_to_threshold" => interpolated_crossing(&time, &values, threshold.unwrap()),
            _ => return Err("unknown scenario ranking metric"),
        };
        if score.is_some_and(|value| !value.is_finite()) {
            return Err("scenario ranking score exceeded finite numerical domain");
        }
        grid = Some(time);
        scores.push((id, score));
    }
    scores.sort_by(|a, b| match (a.1, b.1) {
        (Some(a), Some(b)) if threshold.is_some() => a.partial_cmp(&b).unwrap(),
        (Some(a), Some(b)) => b.partial_cmp(&a).unwrap(),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    let mut place = 1;
    let rows: Vec<_> = scores
        .iter()
        .enumerate()
        .map(|(index, &(id, score))| {
            if index > 0 && scores[index - 1].1 != score {
                place = index + 1;
            }
            json!({"id":id,"place":score.map(|_| place),(metric):score})
        })
        .collect();
    let mut ranking = json!({"metric":metric,"rows":rows,"error":null});
    if let Some(threshold) = threshold {
        ranking["threshold"] = json!(threshold);
    }
    add_time_basis(&mut ranking, basis);
    Ok(ranking)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Comparison {
    baseline: String,
    alternative: String,
    threshold: Option<f64>,
    #[serde(default)]
    interpolate_threshold: bool,
    relative_margin: Option<f64>,
    #[serde(default)]
    compare_resources: bool,
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
    #[serde(default)]
    require_distinct_pair_seeds: bool,
    interval_mass: f64,
    evaluation_time: Option<f64>,
    evaluation_metric: Option<String>,
    #[serde(default)]
    include_path_summary: bool,
    #[serde(default)]
    include_path_covariance: bool,
    bootstrap: Option<EnsembleBootstrap>,
    #[serde(default)]
    exact_sign_flip: bool,
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
    if result["status"] == "paused" {
        return Err("comparison requires a completed simulation; resume the paused scenario");
    }
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
    if comparison.interpolate_threshold && comparison.threshold.is_none() {
        return Err("interpolate_threshold requires a threshold");
    }
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
    if comparison.compare_resources {
        let baseline = paths[comparison.baseline.as_str()].1;
        let alternative = paths[comparison.alternative.as_str()].1;
        result["resource_differences"] = compare_resource_totals(baseline, alternative)?;
    }
    let last = time.len() - 1;
    let (baseline_final, alternative_final) = (baseline[last], alternative[last]);
    result["ranking"] = json!({"metric":"final_adoption",
        "baseline_rank":baseline_final,"alternative_rank":alternative_final,
        "baseline_place":1 + usize::from(baseline_final < alternative_final),
        "alternative_place":1 + usize::from(alternative_final < baseline_final),
        "tie":baseline_final == alternative_final});
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
        if comparison.interpolate_threshold {
            result["interpolated_threshold_timing"] = json!({
                "method":"linear_between_samples",
                "baseline":interpolated_crossing(&time, &baseline, threshold),
                "alternative":interpolated_crossing(&time, &alternative, threshold)
            });
        }
    }
    Ok(result)
}

const RESOURCE_TOTAL_FIELDS: &[&str] = &[
    "occupied_slot_ticks",
    "configured_slot_ticks",
    "idle_slot_ticks",
    "over_capacity_slot_ticks",
    "admitted_wait_ticks",
    "abandoned_wait_ticks",
    "pending_wait_ticks",
    "peak_queue_length",
    "peak_in_service",
];
const RESOURCE_TRACE_FIELDS: &[(&str, &str)] = &[
    ("completions", "completed_services"),
    ("balks", "balked_requests"),
    ("abandonments", "abandoned_requests"),
];
const RESOURCE_TERMINAL_FIELDS: &[(&str, &str)] = &[
    ("queue_length", "waiting_at_horizon"),
    ("in_service", "in_service_at_horizon"),
];

fn compare_resource_totals(baseline: &Value, alternative: &Value) -> Result<Value, &'static str> {
    let baseline_metrics = &baseline["resource_metrics"];
    let alternative_metrics = &alternative["resource_metrics"];
    if !baseline_metrics.is_object() || !alternative_metrics.is_object() {
        return Err("resource comparison requires two Kairos network resource results");
    }
    let ticks_per_unit = &baseline["metadata"]["ticks_per_unit"];
    if ticks_per_unit != &alternative["metadata"]["ticks_per_unit"]
        || baseline_metrics["horizon_ticks"] != alternative_metrics["horizon_ticks"]
    {
        return Err("resource comparison requires matching tick resolution and horizon");
    }
    let mut result = json!({"ticks_per_unit":ticks_per_unit,
        "horizon_ticks":baseline_metrics["horizon_ticks"]});
    for &key in RESOURCE_TOTAL_FIELDS {
        let read = |metrics: &Value| metrics[key].as_i64().ok_or("invalid resource metric");
        result[key] = json!(read(alternative_metrics)? - read(baseline_metrics)?);
    }
    for &(field, metric) in RESOURCE_TRACE_FIELDS {
        let count = |outcome: &Value| {
            outcome[field]
                .as_array()
                .map(|events| events.len() as i64)
                .ok_or("invalid resource event trace")
        };
        result[metric] = json!(count(alternative)? - count(baseline)?);
    }
    for &(field, metric) in RESOURCE_TERMINAL_FIELDS {
        let count = |outcome: &Value| {
            outcome["metadata"][field]
                .as_i64()
                .ok_or("invalid terminal resource count")
        };
        result[metric] = json!(count(alternative)? - count(baseline)?);
    }
    Ok(result)
}

fn interpolated_crossing(time: &[f64], values: &[f64], threshold: f64) -> Option<f64> {
    let index = values.iter().position(|&value| value >= threshold)?;
    if index == 0 || values[index] == threshold {
        return Some(time[index]);
    }
    let fraction = threshold_fraction(values[index - 1], threshold, values[index]);
    let lower = time[index - 1];
    let upper = time[index];
    let span = upper - lower;
    Some(if !span.is_finite() {
        lower * (1. - fraction) + upper * fraction
    } else if fraction <= 0.5 {
        fraction.mul_add(span, lower)
    } else {
        (1. - fraction).mul_add(-span, upper)
    })
}

fn threshold_fraction(lower: f64, threshold: f64, upper: f64) -> f64 {
    let span = upper - lower;
    if span.is_finite() {
        return (threshold - lower) / span;
    }

    let scale = lower.abs().max(upper.abs());
    (threshold / scale - lower / scale) / (upper / scale - lower / scale)
}

#[cfg(test)]
mod tests {
    use super::{
        adoption_area, centered_offsets, ensemble_moments, exact_sign_flip, threshold_fraction,
    };

    #[test]
    fn resource_inference_rejects_unrepresentable_integer_effects() {
        let exact = 1_i64 << 53;
        assert_eq!(super::exact_resource_effect(exact), Ok(exact as f64));
        assert_eq!(
            super::exact_resource_effect(exact + 2),
            Ok((exact + 2) as f64)
        );
        assert!(super::exact_resource_effect(exact + 1).is_err());
        assert!(super::exact_resource_effect(-exact - 1).is_err());
    }

    #[test]
    fn threshold_crossing_preserves_subnormal_time_and_overflowing_spans() {
        let tiny = f64::from_bits(1);
        assert_eq!(
            super::interpolated_crossing(&[tiny, 2. * tiny], &[0., 2.], 1.),
            Some(2. * tiny)
        );
        assert_eq!(
            super::interpolated_crossing(&[-f64::MAX, f64::MAX], &[0., 2.], 1.),
            Some(0.)
        );
    }

    #[test]
    fn adoption_area_preserves_finite_extreme_trapezoids() {
        let tiny = f64::from_bits(1);
        assert_eq!(adoption_area(&[0., 1.], &[tiny, tiny]), Ok(tiny));
        assert_eq!(adoption_area(&[0., 2.], &[0., tiny]), Ok(tiny));
        assert_eq!(
            adoption_area(&[-f64::MAX, f64::MAX], &[0., tiny]),
            Ok(f64::MAX * tiny)
        );
        let result = adoption_area(&[-f64::MAX, f64::MAX], &[0.25, 0.25]);
        assert_eq!(result, Ok(f64::MAX * 0.5));
        assert!(adoption_area(&[-f64::MAX, f64::MAX], &[1., 1.]).is_err());
    }

    #[test]
    fn ensemble_standard_error_resolves_adjacent_effects() {
        let lower = 1.5_f64;
        let upper = lower.next_up();
        let (_, error) = ensemble_moments(&[lower, upper], upper);
        assert!((error / ((upper - lower) / 2.) - 1.).abs() <= 2. * f64::EPSILON);
        let variance = centered_offsets(&[lower, upper], upper)
            .map(|offset| ((offset / 2_f64.sqrt()) * upper).powi(2))
            .sum::<f64>();
        assert!((variance / error.powi(2) - 1.).abs() <= 4. * f64::EPSILON);
        let (mean, error) = ensemble_moments(&[-f64::MAX, f64::MAX], f64::MAX);
        assert_eq!(mean, 0.);
        assert_eq!(error, f64::MAX);
    }

    #[test]
    fn exact_sign_flip_keeps_tied_assignments() {
        let result = exact_sign_flip(&[-10., -6., 10.], 10.).unwrap();
        assert_eq!(result["extreme_assignments"], 8);
        assert_eq!(result["p_value"], 1.);
    }

    #[test]
    fn exact_sign_flip_calibrates_under_exchangeable_signs() {
        let magnitudes = [1., 2., 4., 8., 16., 32.];
        let mut p_values: Vec<_> = (0..64)
            .map(|assignment| {
                let effects = magnitudes.map(|magnitude| {
                    if assignment & magnitude as usize == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                });
                exact_sign_flip(&effects, 32.).unwrap()["p_value"]
                    .as_f64()
                    .unwrap()
            })
            .collect();
        p_values.sort_unstable_by(f64::total_cmp);
        for (rank, pair) in p_values.as_chunks::<2>().0.iter().enumerate() {
            assert_eq!(pair, &[((rank + 1) as f64) / 32.; 2]);
        }
        assert_eq!(p_values.iter().filter(|&&p| p <= 0.05).count(), 2);
    }

    #[test]
    fn threshold_fraction_preserves_a_narrow_bracket() {
        let lower = 1.5_f64;
        let threshold = lower.next_up();
        let upper = threshold.next_up();
        assert_eq!(threshold_fraction(lower, threshold, upper), 0.5);
    }

    #[test]
    fn threshold_fraction_handles_an_overflowing_span() {
        assert_eq!(threshold_fraction(-f64::MAX, 0., f64::MAX), 0.5);
    }
}

fn adoption_area(time: &[f64], values: &[f64]) -> Result<f64, &'static str> {
    if time.len() < 2 {
        return Err("adoption area requires at least two observations");
    }
    let area = time
        .windows(2)
        .zip(values.windows(2))
        .map(|(t, y)| {
            let sum = y[0] + y[1];
            let width = t[1] - t[0];
            if sum.is_subnormal() && width >= 2. {
                // Scale the time span first so an odd subnormal sum keeps its low bit.
                let half_width = if width.is_finite() {
                    width * 0.5
                } else {
                    t[1] * 0.5 - t[0] * 0.5
                };
                return half_width * sum;
            }
            let height = if sum.is_finite() {
                sum * 0.5
            } else {
                0.5 * y[0] + 0.5 * y[1]
            };
            if width.is_finite() {
                width * height
            } else {
                t[1] * height - t[0] * height
            }
        })
        .sum::<f64>();
    area.is_finite()
        .then_some(area)
        .ok_or("adoption area exceeded finite numerical domain")
}

fn exact_resource_effect(difference: i64) -> Result<f64, &'static str> {
    let effect = difference as f64;
    if effect as i128 != i128::from(difference) {
        return Err("ensemble resource difference cannot be represented exactly");
    }
    Ok(effect)
}

fn compare_ensemble(
    paths: &ScenarioPaths<'_>,
    comparison: &EnsembleComparison,
) -> Result<Value, &'static str> {
    let mass = comparison.interval_mass;
    let resource_metric = comparison
        .evaluation_metric
        .as_deref()
        .and_then(|metric| metric.strip_prefix("resource."));
    if resource_metric.is_some() && comparison.evaluation_time.is_some() {
        return Err("resource metrics cannot be combined with evaluation_time");
    }
    let mut resource_basis = None;
    let compare_area = match comparison.evaluation_metric.as_deref() {
        None => false,
        Some("adoption_area") if comparison.evaluation_time.is_none() => true,
        Some("adoption_area") => {
            return Err("adoption_area cannot be combined with evaluation_time");
        }
        Some(_) if resource_metric.is_some() => false,
        Some(_) => return Err("unknown ensemble evaluation metric"),
    };
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
    let mut pair_seeds = HashSet::new();
    let mut grid = None;
    let mut endpoint = None;
    let mut differences = Vec::with_capacity(comparison.pairs.len());
    let mut path_differences = Vec::<Vec<f64>>::new();
    let summarize_path = comparison.include_path_summary || comparison.include_path_covariance;
    for pair in &comparison.pairs {
        if !used.insert(&pair.baseline) || !used.insert(&pair.alternative) {
            return Err("each scenario ID can appear only once per ensemble");
        }
        let (time, baseline) = scalar_path(paths, &pair.baseline)?;
        let (alternative_time, alternative) = scalar_path(paths, &pair.alternative)?;
        if time != alternative_time || grid.as_ref().is_some_and(|expected| expected != &time) {
            return Err("ensemble comparisons require identical time grids across all replicates");
        }
        if summarize_path {
            if comparison.include_path_covariance && time.len() > 128 {
                return Err("ensemble path covariance supports at most 128 sample times");
            }
            if time.len() > 1_000_000 / comparison.pairs.len() {
                return Err("ensemble path summary exceeds 1000000 paired samples");
            }
            if path_differences.is_empty() {
                path_differences.resize_with(time.len(), Vec::new);
            }
            for ((values, &alternative), &baseline) in
                path_differences.iter_mut().zip(&alternative).zip(&baseline)
            {
                let difference = alternative - baseline;
                if !difference.is_finite() {
                    return Err("ensemble path difference must be finite");
                }
                values.push(difference);
            }
        }
        if comparison.require_distinct_pair_seeds {
            let seed = |id: &str| {
                paths
                    .get(id)
                    .and_then(|(_, result)| result["metadata"]["seed"].as_u64())
            };
            let baseline_seed = seed(&pair.baseline).ok_or(
                "seed validation requires an effective seed for both scenarios in each pair",
            )?;
            if seed(&pair.alternative) != Some(baseline_seed) {
                return Err("seed validation requires the same effective seed within each pair");
            }
            if !pair_seeds.insert(baseline_seed) {
                return Err("seed validation requires a distinct effective seed for every pair");
            }
        }
        let index = match comparison.evaluation_time {
            Some(target) => time
                .iter()
                .position(|&sample| sample == target)
                .ok_or("evaluation_time must match a sample on the shared time grid")?,
            None => time.len() - 1,
        };
        let difference = if let Some(metric) = resource_metric {
            let totals = compare_resource_totals(
                paths[pair.baseline.as_str()].1,
                paths[pair.alternative.as_str()].1,
            )?;
            let units = (
                totals["ticks_per_unit"].clone(),
                totals["horizon_ticks"].clone(),
            );
            if resource_basis.as_ref().is_some_and(|basis| basis != &units) {
                return Err(
                    "resource ensemble requires matching tick resolution and horizon across pairs",
                );
            }
            resource_basis = Some(units);
            if matches!(metric, "ticks_per_unit" | "horizon_ticks") {
                return Err("unknown ensemble resource metric");
            }
            let difference = totals[metric]
                .as_i64()
                .ok_or("unknown ensemble resource metric")?;
            exact_resource_effect(difference)?
        } else if compare_area {
            let differences: Vec<_> = alternative
                .iter()
                .zip(&baseline)
                .map(|(a, b)| a - b)
                .collect();
            adoption_area(&time, &differences)?
        } else {
            endpoint = Some(time[index]);
            alternative[index] - baseline[index]
        };
        grid.get_or_insert(time);
        if !difference.is_finite() {
            return Err("ensemble difference must be finite");
        }
        differences.push(difference);
    }
    // Scale before squaring so finite effects near the floating-point limits
    // can still yield a representable standard error.
    let scale = differences.iter().map(|x| x.abs()).fold(0., f64::max);
    let (mean, standard_error) = ensemble_moments(&differences, scale);
    let mut sorted = differences.clone();
    sorted.sort_unstable_by(f64::total_cmp);
    let tail = (1. - mass) / 2.;
    let method = if resource_metric.is_some() {
        "paired_empirical_resource_difference"
    } else if compare_area {
        "paired_empirical_adoption_area_difference"
    } else if comparison.evaluation_time.is_some() {
        "paired_empirical_sample_difference"
    } else {
        "paired_empirical_final_difference"
    };
    let mut result = json!({"method":method,"quantile_method":"linear","evaluation_time":endpoint,
        "pair_count":differences.len(),"differences":differences,"interval_mass":mass,
        "mean_difference":mean,"mean_standard_error":standard_error,
        "standard_error_assumption":"independent replicate pairs",
        "median_difference":linear_quantile(&sorted, 0.5),"empirical_interval":[linear_quantile(&sorted, tail),linear_quantile(&sorted, 1.-tail)]});
    if let Some((ticks_per_unit, horizon_ticks)) = resource_basis {
        result["resource_metric"] = json!(resource_metric);
        result["ticks_per_unit"] = ticks_per_unit;
        result["horizon_ticks"] = horizon_ticks;
    }
    if compare_area {
        let time = grid.as_ref().unwrap();
        result["evaluation_horizon"] = json!([time[0], time[time.len() - 1]]);
    }
    if comparison.include_path_covariance {
        let count = differences.len() as f64;
        let divisor = (count * (count - 1.)).sqrt();
        let centered: Vec<Vec<f64>> = path_differences
            .iter()
            .map(|values| {
                let scale = values.iter().map(|x| x.abs()).fold(0., f64::max);
                centered_offsets(values, scale)
                    .map(|offset| (offset / divisor) * scale)
                    .collect()
            })
            .collect();
        let mut covariance = vec![vec![0.; centered.len()]; centered.len()];
        for (i, left) in centered.iter().enumerate() {
            for (j, right) in centered.iter().enumerate().take(i + 1) {
                let value = left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
                if !value.is_finite() {
                    return Err("ensemble path covariance exceeded finite numerical domain");
                }
                covariance[i][j] = value;
                covariance[j][i] = value;
            }
        }
        result["path_covariance"] = json!(covariance);
    }
    if summarize_path {
        let rows: Vec<_> = grid.as_ref().unwrap().iter().zip(path_differences).map(|(&time, mut values)| {
            let scale = values.iter().map(|x| x.abs()).fold(0., f64::max);
            let (mean, error) = ensemble_moments(&values, scale);
            values.sort_unstable_by(f64::total_cmp);
            json!({"time":time,"mean_difference":mean,"mean_standard_error":error,
                "empirical_interval":[linear_quantile(&values, tail),linear_quantile(&values, 1.-tail)]})
        }).collect();
        result["path_summary"] = json!({"method":"pointwise_paired_empirical","rows":rows});
    }
    if comparison.exact_sign_flip {
        result["sign_flip"] = exact_sign_flip(&differences, scale)?;
    }
    if let Some(options) = &comparison.bootstrap {
        result["bootstrap"] = bootstrap_mean(&differences, options, scale, mass)?;
    }
    if comparison.require_distinct_pair_seeds {
        result["seed_design"] = json!("matched_within_pair_distinct_between_pairs");
    }
    add_time_basis(&mut result, basis);
    Ok(result)
}

fn ensemble_moments(differences: &[f64], scale: f64) -> (f64, f64) {
    if scale == 0. {
        return (0., 0.);
    }
    let n = differences.len() as f64;
    let mean = differences.iter().map(|x| x / scale).sum::<f64>() / n;
    let squares = centered_offsets(differences, scale)
        .map(|offset| offset.powi(2))
        .sum::<f64>();
    (mean * scale, (squares / (n * (n - 1.))).sqrt() * scale)
}

fn centered_offsets(differences: &[f64], scale: f64) -> impl Iterator<Item = f64> + '_ {
    // Subtract the anchor before scaling to retain differences of a few ULPs.
    let scale = if scale == 0. { 1. } else { scale };
    let anchor = differences[0];
    let offset = move |&x: &f64| {
        let delta = x - anchor;
        if delta.is_finite() {
            delta / scale
        } else {
            x / scale - anchor / scale
        }
    };
    let center = differences.iter().map(offset).sum::<f64>() / differences.len() as f64;
    differences.iter().map(move |x| offset(x) - center)
}

fn exact_sign_flip(differences: &[f64], scale: f64) -> Result<Value, &'static str> {
    if differences.len() > 16 {
        return Err("exact sign-flip inference supports at most 16 pairs");
    }
    // Scale by a power of two to avoid rounding each input differently
    // and breaking ties between otherwise equivalent sign assignments.
    let exponent = scale.to_bits() & f64::INFINITY.to_bits();
    let divisor = f64::from_bits(exponent).max(f64::MIN_POSITIVE);
    let values: Vec<_> = differences.iter().map(|x| x / divisor).collect();
    let observed = values.iter().sum::<f64>().abs();
    let assignments = 1usize << values.len();
    let extreme = (0..assignments)
        .filter(|mask| {
            let sum: f64 = values
                .iter()
                .enumerate()
                .map(|(i, &x)| if mask & (1 << i) == 0 { x } else { -x })
                .sum();
            sum.abs() >= observed
        })
        .count();
    Ok(
        json!({"method":"exact_two_sided_sign_flip","assignments":assignments,
        "extreme_assignments":extreme,"p_value":extreme as f64 / assignments as f64,
        "assumption":"independent pairs with exchangeable signs under the null"}),
    )
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
                    "engines":["kairos_network"],"network_types":["custom","ring","random_fixed_edges","scale_free"]
                });
                capability["resource_comparison"] = json!({
                    "option":"compare_resources","engine":"kairos_network",
                    "ensemble_evaluation_metric":"resource.<metric>",
                    "ensemble_metrics":RESOURCE_TOTAL_FIELDS.iter().copied()
                        .chain(RESOURCE_TRACE_FIELDS.iter().chain(RESOURCE_TERMINAL_FIELDS).map(|&(_, metric)| metric))
                        .map(|metric| format!("resource.{metric}")).collect::<Vec<_>>(),
                    "difference":"alternative_minus_baseline",
                    "requires":["shared_observation_grid","shared_tick_resolution","shared_horizon"]
                });
            }
            let event_pause = match family {
                "baseline" | "intervention" => Some("pause_after_adoptions"),
                "network" => Some("pause_after_completions"),
                _ => None,
            };
            if let Some(event_pause) = event_pause {
                capability["checkpoint_resume"] = json!({
                    "pause_options":[event_pause,"pause_at_tick"],
                    "resume_option":"resume_checkpoint",
                    "request_section":"simulation_options",
                    "paused_result_status":"paused"
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
    submitted_simulation: Option<Value>,
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
        if artifact.scenario.is_none()
            || artifact.seed.is_none()
            || artifact.submitted_simulation.is_some()
        {
            return invalid(
                "legacy artifact requires scenario and seed and cannot include submitted simulation inputs",
            );
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

// R serializes whole-valued doubles as JSON integers. Accept that spelling
// only when translation expects a float and the integer is exactly equal.
fn replay_payload_matches(expected: &Value, replay: &Value) -> bool {
    if expected == replay {
        return true;
    }
    match (expected, replay) {
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter().all(|(key, value)| {
                    b.get(key)
                        .is_some_and(|other| replay_payload_matches(value, other))
                })
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| replay_payload_matches(a, b))
        }
        (Value::Number(a), Value::Number(b)) if a.is_f64() => {
            let expected = a.as_f64().unwrap();
            let replay = b
                .as_i64()
                .map(i128::from)
                .or_else(|| b.as_u64().map(i128::from));
            expected.fract() == 0. && replay.is_some_and(|integer| expected as i128 == integer)
        }
        _ => false,
    }
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
    if replay.as_object().is_none_or(|fields| {
        fields
            .keys()
            .any(|key| !matches!(key.as_str(), "schema_version" | "operation" | "payload"))
    }) {
        return Err("artifact replay request contains unsupported fields");
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
    validate_network_horizon(&entry)?;
    let seed_matches = match entry.simulation["simulation_options"].get("seed") {
        Some(seed) => seed.is_u64() && seed == &original["seed"],
        None => artifact.seed.is_none(),
    };
    if !seed_matches {
        return Err("artifact replay seed does not match the envelope");
    }
    if artifact.version != env!("CARGO_PKG_VERSION") {
        return Ok(Some("artifact was created by a different kernel version"));
    }
    if let Some(submitted) = &artifact.submitted_simulation {
        let mut translated = entry;
        let resolved = std::mem::replace(&mut translated.simulation, submitted.clone());
        apply_specification(&mut translated)?;
        if !replay_payload_matches(&translated.simulation, &resolved) {
            return Err("submitted simulation does not resolve to the embedded replay payload");
        }
    }
    Ok(None)
}

#[cfg(test)]
#[test]
fn replay_payload_accepts_only_exact_integer_spellings_of_floats() {
    let expected = json!({"parameters": [1.0, 9_007_199_254_740_992.0]});
    assert!(replay_payload_matches(
        &expected,
        &json!({"parameters": [1, 9_007_199_254_740_992_u64]})
    ));
    assert!(!replay_payload_matches(
        &expected,
        &json!({"parameters": [1, 9_007_199_254_740_993_u64]})
    ));
    assert!(!replay_payload_matches(&json!(1), &json!(1.0)));
}
