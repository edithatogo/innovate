//! Heterogeneous network adoption with Kairos agent state, dispatch and priority service.
use crate::validation::{numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use kairo_ecs_abm::ABMContext;
use kairo_ecs_des::{PriorityResource, RenegingQueue};
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
    #[serde(default)]
    directed: bool,
    #[serde(default)]
    edge_weights: Option<Vec<f64>>,
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
    #[serde(default)]
    agent_priorities: Option<Vec<i32>>,
    max_wait_ticks: Option<u64>,
    max_queue_length: Option<usize>,
    retry: Option<Retry>,

    #[serde(default)]
    schedule: Vec<Change>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Retry {
    delay_ticks: u64,
    max_retries: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Change {
    tick: u64,
    edges: Option<Vec<[usize; 2]>>,
    edge_weights: Option<Vec<f64>>,
    service_ticks: Option<u64>,
    capacity: Option<u64>,
    p: Option<f64>,
    q: Option<f64>,
    innovation_multipliers: Option<Vec<f64>>,
    imitation_multipliers: Option<Vec<f64>>,
}

struct Graph {
    incoming: Vec<Vec<(usize, f64)>>,
    totals: Vec<f64>,
    weighted: bool,
}

impl Graph {
    fn new(
        population: usize,
        directed: bool,
        edges: &[[usize; 2]],
        weights: Option<&[f64]>,
    ) -> Result<Self, &'static str> {
        let mut incoming = vec![Vec::new(); population];
        let mut unique = std::collections::HashSet::new();
        if weights.is_some_and(|weights| {
            weights.len() != edges.len() || weights.iter().any(|w| !w.is_finite() || *w <= 0.)
        }) {
            return Err("edge_weights must contain one finite positive weight per edge");
        }
        for (index, &[a, b]) in edges.iter().enumerate() {
            let key = if directed {
                (a, b)
            } else {
                (a.min(b), a.max(b))
            };
            if a >= population || b >= population || a == b || !unique.insert(key) {
                return Err("edges must be unique non-self pairs within population");
            }
            let weight = weights.map_or(1., |w| w[index]);
            incoming[b].push((a, weight));
            if !directed {
                incoming[a].push((b, weight));
            }
        }
        let mut totals = vec![0.; population];
        if weights.is_some() {
            for (neighbors, total) in incoming.iter_mut().zip(&mut totals) {
                let scale = neighbors.iter().map(|(_, w)| *w).fold(0., f64::max);
                for (_, weight) in neighbors {
                    *weight /= scale;
                    *total += *weight;
                }
            }
        }
        Ok(Self {
            incoming,
            totals,
            weighted: weights.is_some(),
        })
    }
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
        || o.max_wait_ticks
            .is_some_and(|ticks| !(1..=1_000_000_000_000).contains(&ticks))
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
    if o.max_queue_length.is_some_and(|limit| limit > o.population)
        || o.retry.as_ref().is_some_and(|retry| {
            !(1..=1_000_000_000_000).contains(&retry.delay_ticks)
                || !(1..=16).contains(&retry.max_retries)
        })
    {
        return Err(invalid("invalid queue length or retry policy"));
    }
    let mut innovation_weights = o
        .innovation_multipliers
        .clone()
        .unwrap_or_else(|| vec![1.; o.population]);
    let mut imitation_weights = o
        .imitation_multipliers
        .clone()
        .unwrap_or_else(|| vec![1.; o.population]);
    let service = o
        .agent_service_ticks
        .clone()
        .unwrap_or_else(|| vec![o.service_ticks; o.population]);
    let priorities = o.agent_priorities.unwrap_or_else(|| vec![0; o.population]);
    if priorities.len() != o.population {
        return Err(invalid(
            "agent_priorities must be a population-length array of signed 32-bit integers",
        ));
    }
    let valid_multipliers = |values: &[f64]| {
        values.len() == o.population && values.iter().all(|v| v.is_finite() && *v >= 0.)
    };
    if !valid_multipliers(&innovation_weights)
        || !valid_multipliers(&imitation_weights)
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
    let mut innovation = required_f64(p, "p", op)?;
    let mut imitation = required_f64(p, "q", op)?;
    let market = required_f64(p, "m", op)?;
    let mut graph = Graph::new(
        o.population,
        o.directed,
        &o.edges,
        o.edge_weights.as_deref(),
    )
    .map_err(invalid)?;
    if o.schedule.len() > 64
        || o.schedule.windows(2).any(|w| w[0].tick >= w[1].tick)
        || o.schedule
            .iter()
            .map(|c| c.edges.as_ref().map_or(0, Vec::len))
            .sum::<usize>()
            > 1_000_000
    {
        return Err(invalid(
            "schedule requires at most 64 strictly increasing changes and at most 1000000 replacement edges",
        ));
    }
    let changes = o
        .schedule
        .iter()
        .map(|change| {
            if change.tick == 0
                || change.tick > 1_000_000_000_000_000
                || (change.edges.is_none()
                    && (change.edge_weights.is_some()
                        || (change.service_ticks.is_none()
                            && change.capacity.is_none()
                            && change.p.is_none()
                            && change.q.is_none()
                            && change.innovation_multipliers.is_none()
                            && change.imitation_multipliers.is_none())))
                || [change.p, change.q]
                    .into_iter()
                    .flatten()
                    .any(|rate| !rate.is_finite() || rate < 0.)
                || [change.innovation_multipliers.as_deref(), change.imitation_multipliers.as_deref()]
                    .into_iter()
                    .flatten()
                    .any(|values| !valid_multipliers(values))
                || change.capacity.is_some_and(|cap| cap > o.population as u64)
                || change
                    .service_ticks
                    .is_some_and(|ticks| !(1..=1_000_000_000_000).contains(&ticks))
            {
                return Err(invalid(
                    "invalid scheduled tick, graph, capacity, hazard, agent multipliers or service duration",
                ));
            }
            let replacement = change
                .edges
                .as_ref()
                .map(|edges| {
                    Graph::new(
                        o.population,
                        o.directed,
                        edges,
                        change.edge_weights.as_deref(),
                    )
                    .map_err(invalid)
                })
                .transpose()?;
            Ok((u128::from(change.tick), replacement, change))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut changes = changes.into_iter().peekable();
    let mut applied_ticks = Vec::new();
    let mut service_override = None;
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
    // Kairos orders waiters. Admission is separate because its priority resource
    // has fixed capacity and active work must survive a capacity decrease.
    let mut resource = PriorityResource::new("adoption_waiters", 0);
    let mut capacity = o.capacity;
    let mut in_service = 0_u64;
    let mut reneging = RenegingQueue::new();
    let mut deadlines = vec![None; o.population];
    let mut abandoned = Vec::new();
    let mut balks = Vec::new();
    let mut retries = Vec::new();
    let mut retry_at = vec![None; o.population];
    let mut retry_counts = vec![0; o.population];
    let next_retry = |index: usize, now: u128, counts: &mut [u8]| {
        let policy = o.retry.as_ref()?;
        if counts[index] >= policy.max_retries {
            return None;
        }
        counts[index] += 1;
        Some(now + u128::from(policy.delay_ticks))
    };
    let mut waiting_since = vec![None; o.population];
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
                    let degree = graph.incoming[i].len();
                    innovation * innovation_weights[i]
                        + if degree == 0 {
                            0.
                        } else if graph.weighted {
                            let exposed: f64 = graph.incoming[i]
                                .iter()
                                .filter(|(j, _)| context.get::<u8>(agents[*j]) == Some(&2))
                                .map(|(_, weight)| weight)
                                .sum();
                            imitation * imitation_weights[i] * (exposed / graph.totals[i])
                        } else {
                            imitation
                                * imitation_weights[i]
                                * graph.incoming[i]
                                    .iter()
                                    .filter(|(j, _)| context.get::<u8>(agents[*j]) == Some(&2))
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
        let boundary = changes
            .peek()
            .map(|(tick, _, _)| *tick)
            .filter(|t| *t <= horizon);
        let expiry = deadlines
            .iter()
            .flatten()
            .copied()
            .min()
            .filter(|t| *t <= horizon);
        let wake = retry_at
            .iter()
            .flatten()
            .copied()
            .min()
            .filter(|t| *t <= horizon);
        let process_completion = completion.is_some_and(|t| {
            attempt.is_none_or(|a| t <= a)
                && boundary.is_none_or(|b| t <= b)
                && expiry.is_none_or(|e| t <= e)
                && wake.is_none_or(|w| t <= w)
        });
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
            in_service -= 1;
        } else if let Some(tick) = expiry.filter(|t| {
            boundary.is_none_or(|b| *t <= b)
                && attempt.is_none_or(|a| *t <= a)
                && wake.is_none_or(|w| *t <= w)
        }) {
            now = tick;
            for id in reneging.expire_reneged(SimTime::from_ticks(now)) {
                if deadlines[id.index as usize].take().is_some() {
                    let index = id.index as usize;
                    retry_at[index] = next_retry(index, now, &mut retry_counts);
                    context.attach(
                        id,
                        if retry_at[index].is_some() {
                            4_u8
                        } else {
                            3_u8
                        },
                    );
                    abandoned.push(json!({"tick":now,"agent":id.index}));
                    waiting_since[index] = None;
                }
            }
            let mut waiting: Vec<_> = waiting_since
                .iter()
                .enumerate()
                .filter_map(|(index, tick)| tick.map(|tick| (tick, index)))
                .collect();
            waiting.sort_unstable();
            resource = PriorityResource::new("adoption_waiters", 0);
            for (tick, index) in waiting {
                resource.request(agents[index], priorities[index], SimTime::from_ticks(tick));
            }
        } else if let Some(tick) =
            boundary.filter(|t| attempt.is_none_or(|a| *t <= a) && wake.is_none_or(|w| *t <= w))
        {
            let (_, replacement, change) = changes.next().expect("boundary comes from schedule");
            now = tick;
            if let Some(replacement) = replacement {
                graph = replacement;
            }
            if change.service_ticks.is_some() {
                service_override = change.service_ticks;
            }
            capacity = change.capacity.unwrap_or(capacity);
            innovation = change.p.unwrap_or(innovation);
            imitation = change.q.unwrap_or(imitation);
            if let Some(values) = &change.innovation_multipliers {
                innovation_weights.clone_from(values);
            }
            if let Some(values) = &change.imitation_multipliers {
                imitation_weights.clone_from(values);
            }
            applied_ticks.push(tick);
        } else if let Some(tick) = wake.filter(|t| attempt.is_none_or(|a| *t <= a)) {
            now = tick;
            for (index, at) in retry_at.iter_mut().enumerate() {
                if *at == Some(tick) {
                    *at = None;
                    context.attach(agents[index], 0_u8);
                    retries.push(json!({"tick":now,"agent":agents[index].index}));
                }
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
            if in_service >= capacity
                && o.max_queue_length
                    .is_some_and(|limit| resource.queue_length() >= limit)
            {
                retry_at[index] = next_retry(index, now, &mut retry_counts);
                context.attach(
                    id,
                    if retry_at[index].is_some() {
                        4_u8
                    } else {
                        3_u8
                    },
                );
                balks.push(json!({"tick":now,"agent":id.index}));
                continue;
            }
            context.attach(id, 1_u8);
            waiting_since[index] = Some(now);
            resource.request(id, priorities[index], SimTime::from_ticks(now));
            if let Some(wait) = o.max_wait_ticks.filter(|_| in_service >= capacity) {
                deadlines[index] = Some(now + u128::from(wait));
                reneging.request_with_deadline(
                    id,
                    SimTime::from_ticks(now),
                    SimTime::from_ticks(u128::from(wait)),
                );
            }
        } else {
            break;
        }
        while in_service < capacity {
            let Some(next) = resource.release() else {
                break;
            };
            let index = next.index as usize;
            waiting_since[index] = None;
            deadlines[index] = None;
            context.schedule_behaviour(
                next,
                2,
                SimTime::from_ticks(now + u128::from(service_override.unwrap_or(service[index]))),
            );
            in_service += 1;
        }
    }
    let queue_length = resource.queue_length();
    let available = capacity.saturating_sub(in_service);
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
    let states:Vec<_>=agents.iter().map(|id|json!({"index":id.index,"generation":id.generation,"state":match context.get::<u8>(*id){Some(0)=>"susceptible",Some(1)=>"pending",Some(3)=>"withdrawn",Some(4)=>"cooldown",_=>"adopted"}})).collect();
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        error: None,
        result: Some(
            json!({"values":values,"counts":counts,"agents":states,"requests":requests,"completions":trace,"abandonments":abandoned,"balks":balks,"retry_eligibility":retries,"metadata":{"engine":"kairos_network","kairos_revision":crate::simulation::REV,"seed":o.seed,"population":o.population,"capacity":o.capacity,"current_capacity":capacity,"current_p":innovation,"current_q":imitation,"in_service":in_service,"service_ticks":o.service_ticks,"ticks_per_unit":o.ticks_per_unit,"queue_length":queue_length,"available":available,"agent_priorities":priorities,"max_wait_ticks":o.max_wait_ticks,"max_queue_length":o.max_queue_length,"innovation_multipliers":innovation_weights,"imitation_multipliers":imitation_weights,"agent_service_ticks":service,"directed":o.directed,"weighted":graph.weighted,"applied_schedule_ticks":applied_ticks,"service_override_ticks":service_override,"hazard":if graph.weighted {"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted incoming weight/total incoming weight), zero imitation without incoming edges"} else {"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted neighbors/degree), zero imitation at degree zero"},"resource_semantics":"Nonpreemptive priority, FIFO ties; adoption occurs at service completion; pending agents do not transmit","rng_checkpoint":rng.checkpoint()}}),
        ),
        metadata: json!({"runtime":"rust_native","engine":"kairos_network"}),
    })
}
