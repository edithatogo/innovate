//! Heterogeneous network adoption with Kairos agent state, dispatch and FIFO service.
use crate::validation::{numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use kairo_ecs_abm::ABMContext;
use kairo_ecs_des::Resource;
use kairo_ecs_rng::{
    DeterministicStream,
    rand_distr::{Exp, Uniform},
};
use kairo_ecs_types::{EventKind, SimTime, StepOutcome};
use serde::Deserialize;
use serde_json::json;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    engine: String,
    seed: u64,
    population: usize,
    edges: Vec<[usize; 2]>,
    capacity: u64,
    service_ticks: u64,
    #[serde(default = "ticks")]
    ticks_per_unit: u64,
    #[serde(default)]
    initial_agents: Vec<usize>,
    #[serde(default)]
    innovation_multipliers: Option<Vec<f64>>,
    #[serde(default)]
    imitation_multipliers: Option<Vec<f64>>,
    #[serde(default)]
    agent_service_ticks: Option<Vec<u64>>,
}
fn ticks() -> u64 {
    1000000
}
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = r.operation;
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
    if op != KernelOperation::SimulateModel || r.model_key.as_deref() != Some("bass") {
        return Err(invalid("network engine requires Bass simulate_model"));
    }
    let payload = r
        .payload
        .as_object()
        .ok_or_else(|| invalid("payload must be an object"))?;
    let o: Options = serde_json::from_value(payload["simulation_options"].clone())
        .map_err(|e| invalid(&e.to_string()))?;
    if o.engine != "kairos_network"
        || o.population == 0
        || o.population > 1000
        || o.capacity == 0
        || o.capacity > o.population as u64
        || o.service_ticks == 0
        || o.service_ticks > 1000000000000
        || o.ticks_per_unit == 0
        || o.ticks_per_unit > 1000000000
    {
        return Err(invalid(
            "invalid engine/population/capacity/service/tick bounds",
        ));
    }
    let innovation_weights = o
        .innovation_multipliers
        .clone()
        .unwrap_or_else(|| vec![1.; o.population]);
    let imitation_weights = o
        .imitation_multipliers
        .clone()
        .unwrap_or_else(|| vec![1.; o.population]);
    let service = o
        .agent_service_ticks
        .clone()
        .unwrap_or_else(|| vec![o.service_ticks; o.population]);
    if innovation_weights.len() != o.population
        || imitation_weights.len() != o.population
        || innovation_weights
            .iter()
            .chain(&imitation_weights)
            .any(|v| !v.is_finite() || *v < 0.)
        || service.len() != o.population
        || service.iter().any(|v| *v == 0 || *v > 1000000000000)
    {
        return Err(invalid(
            "agent multipliers must be finite nonnegative population-length arrays; service ticks must be population-length integers in 1..=1000000000000",
        ));
    }
    let input = object_section(payload, "inputs", op)?;
    let time = numeric_array_from_aliases(input, &["time", "t"], op)?;
    if time
        .iter()
        .any(|t| !t.is_finite() || *t < 0. || *t * o.ticks_per_unit as f64 > 1e15)
    {
        return Err(invalid("network time exceeds finite tick domain"));
    }
    let p = payload["state"]["parameters"]
        .as_object()
        .ok_or_else(|| invalid("parameters required"))?;
    if p.len() != 3 {
        return Err(invalid("exactly p/q/m required"));
    }
    let innovation = required_f64(p, "p", op)?;
    let imitation = required_f64(p, "q", op)?;
    let market = required_f64(p, "m", op)?;
    let mut adjacency = vec![Vec::new(); o.population];
    let mut unique = std::collections::HashSet::new();
    for [a, b] in &o.edges {
        if *a >= o.population
            || *b >= o.population
            || a == b
            || !unique.insert((a.min(b), a.max(b)))
        {
            return Err(invalid(
                "edges must be unique undirected non-self pairs within population",
            ));
        }
        adjacency[*a].push(*b);
        adjacency[*b].push(*a);
    }
    let mut initial = std::collections::HashSet::new();
    if o.initial_agents
        .iter()
        .any(|i| *i >= o.population || !initial.insert(*i))
    {
        return Err(invalid("initial agent indices must be unique and in range"));
    }
    let mut context = ABMContext::new(o.seed);
    let agents: Vec<_> = (0..o.population)
        .map(|i| {
            let id = context.spawn_agent();
            context.attach(id, if initial.contains(&i) { 2_u8 } else { 0_u8 });
            id
        })
        .collect();
    let mut resource = Resource::new("adoption_service", o.capacity);
    let mut rng = DeterministicStream::from_run_seed(o.seed);
    let mut now = 0_u128;
    let horizon = (time.last().unwrap() * o.ticks_per_unit as f64).floor() as u128;
    let mut completed = Vec::new();
    let mut trace = Vec::new();
    let mut requests = Vec::new();
    loop {
        let rates: Vec<_> = (0..o.population)
            .map(|i| {
                if context.get::<u8>(agents[i]) != Some(&0) {
                    0.
                } else {
                    let degree = adjacency[i].len();
                    innovation * innovation_weights[i]
                        + if degree == 0 {
                            0.
                        } else {
                            imitation
                                * imitation_weights[i]
                                * adjacency[i]
                                    .iter()
                                    .filter(|j| context.get::<u8>(agents[**j]) == Some(&2))
                                    .count() as f64
                                / degree as f64
                        }
                }
            })
            .collect();
        let total: f64 = rates.iter().sum();
        if !total.is_finite() {
            return Err(invalid("network hazard exceeded finite domain"));
        }
        let attempt = if total > 0. {
            let delay: f64 = rng.sample(Exp::new(total).map_err(|_| invalid("invalid hazard"))?);
            let delta = (delay * o.ticks_per_unit as f64).ceil().max(1.);
            if delta.is_finite() && delta <= horizon.saturating_sub(now) as f64 {
                Some(now + delta as u128)
            } else {
                None
            }
        } else {
            None
        };
        let completion = context
            .scheduler
            .peek_next()
            .map(|e| e.at.ticks())
            .filter(|t| *t <= horizon);
        let process_completion = completion.is_some_and(|t| attempt.is_none_or(|a| t <= a));
        if process_completion {
            let StepOutcome::Dispatched(event) = context.step() else {
                return Err(invalid("missing completion event"));
            };
            now = event.at.ticks();
            let id = event
                .entity
                .ok_or_else(|| invalid("missing agent identity"))?;
            if context.get::<u8>(id) != Some(&1) {
                return Err(invalid("completion agent was not pending"));
            }
            context.attach(id, 2_u8);
            completed.push(now);
            trace.push(json!({"tick":now,"agent":id.index}));
            if let Some(next) = resource.release() {
                context.schedule_behaviour(
                    next,
                    2,
                    SimTime::from_ticks(now + service[next.index as usize] as u128),
                );
            }
        } else if let Some(at) = attempt {
            now = at;
            let draw: f64 = rng
                .sample(Uniform::new(0., total).map_err(|_| invalid("invalid selection range"))?);
            let mut sum = 0.;
            let index = rates
                .iter()
                .position(|r| {
                    sum += r;
                    sum > draw
                })
                .or_else(|| rates.iter().rposition(|r| *r > 0.))
                .ok_or_else(|| invalid("no susceptible agent"))?;
            let id = agents[index];
            context.schedule_behaviour(id, 1, SimTime::from_ticks(now));
            let StepOutcome::Dispatched(event) = context.step() else {
                return Err(invalid("missing request event"));
            };
            if event.entity != Some(id) || event.kind != EventKind::Custom(1) {
                return Err(invalid("unexpected request dispatch"));
            }
            requests.push(json!({"tick":now,"agent":id.index}));
            context.attach(id, 1_u8);
            if resource.request(id) {
                context.schedule_behaviour(
                    id,
                    2,
                    SimTime::from_ticks(now + service[index] as u128),
                );
            }
        } else {
            break;
        }
    }
    let counts: Vec<_> = time
        .iter()
        .map(|t| {
            initial.len()
                + completed
                    .partition_point(|tick| *tick <= (*t * o.ticks_per_unit as f64).floor() as u128)
        })
        .collect();
    let values: Vec<_> = counts
        .iter()
        .map(|n| market * (*n as f64 / o.population as f64))
        .collect();
    let states:Vec<_>=agents.iter().map(|id|json!({"index":id.index,"generation":id.generation,"state":match context.get::<u8>(*id){Some(0)=>"susceptible",Some(1)=>"pending",_=>"adopted"}})).collect();
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        error: None,
        result: Some(
            json!({"values":values,"counts":counts,"agents":states,"requests":requests,"completions":trace,"metadata":{"engine":"kairos_network","kairos_revision":crate::simulation::REV,"seed":o.seed,"population":o.population,"capacity":o.capacity,"service_ticks":o.service_ticks,"ticks_per_unit":o.ticks_per_unit,"queue_length":resource.queue_length(),"available":resource.available_count(),"innovation_multipliers":innovation_weights,"imitation_multipliers":imitation_weights,"agent_service_ticks":service,"hazard":"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted neighbors/degree), zero imitation at degree zero","resource_semantics":"FIFO; adoption occurs at service completion; pending agents do not transmit","rng_checkpoint":rng.checkpoint()}}),
        ),
        metadata: json!({"runtime":"rust_native","engine":"kairos_network"}),
    })
}
