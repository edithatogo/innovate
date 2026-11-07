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
use kairo_ecs_types::{EntityId, EventKind, SimTime, StepOutcome};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
fn normalize_checkpoint_numbers(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(normalize_checkpoint_numbers),
        Value::Object(values) => values.values_mut().for_each(normalize_checkpoint_numbers),
        Value::Number(number) if number.is_f64() => {
            if let Some(real) = number.as_f64()
                && real.fract() == 0.
                && real.abs() <= (1_u64 << 53) as f64
            {
                *number = (real as i64).into();
            }
        }
        _ => {}
    }
}

#[derive(Deserialize, Serialize)]
struct ServiceStart {
    agent: usize,
    tick: u128,
    request_tick: u128,
    wait_ticks: u128,
    service_ticks: u64,
    scheduled_completion_tick: u128,
}

impl ServiceStart {
    fn schedule_completion(&self, context: &mut ABMContext, agent: EntityId) {
        context.schedule_behaviour(
            agent,
            2,
            SimTime::from_ticks(self.scheduled_completion_tick),
        );
    }
}

#[derive(Deserialize, Serialize)]
struct QueueState {
    waiting_since: Vec<Option<u128>>,
    deadlines: Vec<Option<u128>>,
    retry_at: Vec<Option<u128>>,
    retry_counts: Vec<u8>,
}

impl QueueState {
    fn new(population: usize) -> Self {
        Self {
            waiting_since: vec![None; population],
            deadlines: vec![None; population],
            retry_at: vec![None; population],
            retry_counts: vec![0; population],
        }
    }

    fn ordered_waiters(&self) -> Vec<(u128, usize)> {
        let mut waiting: Vec<_> = self
            .waiting_since
            .iter()
            .enumerate()
            .filter_map(|(index, tick)| tick.map(|tick| (tick, index)))
            .collect();
        waiting.sort_unstable();
        waiting
    }

    fn rebuild_waiters(&self, agents: &[EntityId], priorities: &[i32]) -> PriorityResource {
        let mut resource = PriorityResource::new("adoption_waiters", 0);
        for (tick, index) in self.ordered_waiters() {
            resource.request(agents[index], priorities[index], SimTime::from_ticks(tick));
        }
        resource
    }

    fn rebuild_deadlines(&self, agents: &[EntityId]) -> RenegingQueue {
        let mut reneging = RenegingQueue::new();
        for (arrival, index) in self.ordered_waiters() {
            if let Some(deadline) = self.deadlines[index] {
                reneging.request_with_deadline(
                    agents[index],
                    SimTime::from_ticks(arrival),
                    SimTime::from_ticks(deadline - arrival),
                );
            }
        }
        reneging
    }

    fn schedule_retry(&mut self, policy: Option<Retry>, index: usize, now: u128) {
        self.retry_at[index] = policy.and_then(|policy| {
            if self.retry_counts[index] >= policy.max_retries {
                return None;
            }
            self.retry_counts[index] += 1;
            Some(now + u128::from(policy.delay_ticks))
        });
    }
}

#[derive(Default, Deserialize, Serialize)]
struct ResourceAccounting {
    occupied_slot_ticks: u128,
    over_capacity_slot_ticks: u128,
    last_resource_change: u128,
    peak_queue_length: usize,
    peak_in_service: u64,
    configured_slot_ticks: u128,
    last_capacity_change: u128,
    admitted_wait_ticks: u128,
    abandoned_wait_ticks: u128,
}

impl ResourceAccounting {
    fn record_over_capacity(&mut self, tick: u128, in_service: u64, capacity: u64) {
        self.over_capacity_slot_ticks +=
            (tick - self.last_resource_change) * u128::from(in_service.saturating_sub(capacity));
        self.last_resource_change = tick;
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct NetworkCheckpoint {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pause_tick: Option<u128>,
    configuration: String,
    states: Vec<u8>,
    rng: kairo_ecs_rng::RngCheckpoint,
    now: u128,
    applied_ticks: Vec<u128>,
    service_override: Option<u64>,
    capacity: u64,
    max_wait_ticks: Option<u64>,
    max_queue_length: Option<usize>,
    in_service: u64,
    queue: QueueState,
    service_starts: Vec<ServiceStart>,
    abandoned: Vec<Value>,
    balks: Vec<Value>,
    retries: Vec<Value>,
    retry_policy: Option<Retry>,
    metrics: ResourceAccounting,
    completed: Vec<u128>,
    trace: Vec<Value>,
    requests: Vec<Value>,
    innovation: f64,
    imitation: f64,
    innovation_weights: Vec<f64>,
    imitation_weights: Vec<f64>,
    priorities: Vec<i32>,
    service: Vec<u64>,
}

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
    hazard_forcing: Option<HazardForcing>,

    #[serde(default)]
    schedule: Vec<Change>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Retry {
    delay_ticks: u64,
    max_retries: u8,
}

impl Retry {
    fn invalid(&self) -> bool {
        !(1..=1_000_000_000_000).contains(&self.delay_ticks)
            || !(1..=16).contains(&self.max_retries)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Change {
    tick: u64,
    edges: Option<Vec<[usize; 2]>>,
    edge_weights: Option<Vec<f64>>,
    agent_service_ticks: Option<Vec<u64>>,
    service_ticks: Option<u64>,
    agent_priorities: Option<Vec<i32>>,
    capacity: Option<u64>,
    #[serde(default, deserialize_with = "queue_policy_change")]
    max_wait_ticks: Option<Option<u64>>,
    #[serde(default, deserialize_with = "queue_policy_change")]
    max_queue_length: Option<Option<usize>>,
    #[serde(default, deserialize_with = "queue_policy_change")]
    retry: Option<Option<Retry>>,
    p: Option<f64>,
    q: Option<f64>,
    innovation_multipliers: Option<Vec<f64>>,
    imitation_multipliers: Option<Vec<f64>>,
}

// Omission leaves the policy in place. Explicit null clears the restriction.
fn queue_policy_change<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HazardForcing {
    time: Vec<f64>,
    innovation: Vec<f64>,
    imitation: Vec<f64>,
}

impl HazardForcing {
    fn valid(&self) -> bool {
        (1..=4096).contains(&self.time.len())
            && self.time[0] == 0.
            && self.time.iter().all(|t| t.is_finite())
            && self.time.windows(2).all(|w| w[0] < w[1])
            && [&self.innovation, &self.imitation]
                .into_iter()
                .all(|values| {
                    values.len() == self.time.len()
                        && values.iter().all(|v| v.is_finite() && *v >= 0.)
                })
    }

    fn coefficients(&self, t: f64, p: f64, q: f64) -> (f64, f64) {
        let next = self.time.partition_point(|point| *point <= t);
        let prev = next - 1;
        let interpolate = |values: &[f64]| {
            if next == self.time.len() {
                values[prev]
            } else {
                let f = (t - self.time[prev]) / (self.time[next] - self.time[prev]);
                (1. - f) * values[prev] + f * values[next]
            }
        };
        (
            p * interpolate(&self.innovation),
            q * interpolate(&self.imitation),
        )
    }

    fn attempt(
        &self,
        start: f64,
        end: f64,
        mut budget: f64,
        total: impl Fn(f64) -> f64,
    ) -> Result<Option<f64>, &'static str> {
        let mut left = start;
        let mut initial = total(left);
        for right in self
            .time
            .iter()
            .copied()
            .filter(|t| *t > start && *t < end)
            .chain(std::iter::once(end))
        {
            let final_rate = total(right);
            if !initial.is_finite() || !final_rate.is_finite() {
                return Err("forced network hazard exceeded finite domain");
            }
            let width = right - left;
            let integral = |f: f64| {
                let rate = (1. - f) * initial + f * final_rate;
                initial.midpoint(rate) * (width * f)
            };
            let area = integral(1.);
            if area > 0. && budget <= area {
                let (mut low, mut high) = (0_f64, 1_f64);
                for _ in 0..52 {
                    let mid = low.midpoint(high);
                    if integral(mid) < budget {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                return Ok(Some(left + high * width));
            }
            budget -= area;
            left = right;
            initial = final_rate;
        }
        Ok(None)
    }
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
    let mut options = payload["simulation_options"].clone();
    let options_map = options
        .as_object_mut()
        .ok_or_else(|| invalid("network simulation options must be an object"))?;
    let pause = options_map
        .remove("pause_after_completions")
        .map(|value| {
            value
                .as_u64()
                .filter(|&count| count > 0)
                .ok_or_else(|| invalid("pause_after_completions must be a positive integer"))
        })
        .transpose()?;
    let pause_tick = options_map
        .remove("pause_at_tick")
        .map(|value| {
            value
                .as_u64()
                .map(u128::from)
                .ok_or_else(|| invalid("pause_at_tick must be a nonnegative integer"))
        })
        .transpose()?;
    if pause.is_some() && pause_tick.is_some() {
        return Err(invalid(
            "choose either pause_after_completions or pause_at_tick",
        ));
    }
    let resume = options_map.remove("resume_checkpoint");
    let configuration = (pause.is_some() || pause_tick.is_some() || resume.is_some()).then(|| {
        let mut config = json!([
            crate::simulation::REV,
            options,
            payload["inputs"],
            payload["state"]["parameters"]
        ]);
        normalize_checkpoint_numbers(&mut config);
        Sha256::digest(config.to_string())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    });
    let o: Options = serde_json::from_value(options).map_err(|e| invalid(&e.to_string()))?;
    if o.engine != "kairos_network"
        || o.population == 0
        || o.population > 1000
        || o.max_wait_ticks
            .is_some_and(|ticks| !(1..=1_000_000_000_000).contains(&ticks))
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
        || o.retry.as_ref().is_some_and(Retry::invalid)
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
    let mut service = o
        .agent_service_ticks
        .clone()
        .unwrap_or_else(|| vec![o.service_ticks; o.population]);
    let mut priorities = o.agent_priorities.unwrap_or_else(|| vec![0; o.population]);
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
    if o.hazard_forcing
        .as_ref()
        .is_some_and(|forcing| !forcing.valid())
    {
        return Err(invalid(
            "hazard_forcing requires 1..=4096 increasing finite times starting at zero and finite nonnegative innovation/imitation multipliers",
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
                            && change.agent_service_ticks.is_none()
                            && change.capacity.is_none()
                            && change.agent_priorities.is_none()
                            && change.max_queue_length.is_none()
                            && change.max_wait_ticks.is_none()
                            && change.retry.is_none()
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
                || change.agent_priorities.as_ref().is_some_and(|values| values.len() != o.population)
                || change.agent_service_ticks.as_ref().is_some_and(|values| {
                    change.service_ticks.is_some() || values.len() != o.population
                        || values.iter().any(|ticks| !(1..=1_000_000_000_000).contains(ticks))
                })
                || change.capacity.is_some_and(|cap| cap > o.population as u64)
                || change.max_wait_ticks.flatten().is_some_and(|wait| !(1..=1_000_000_000_000).contains(&wait))
                || change.retry.flatten().as_ref().is_some_and(Retry::invalid)
                || change.max_queue_length.flatten().is_some_and(|limit| limit > o.population)
                || change
                    .service_ticks
                    .is_some_and(|ticks| !(1..=1_000_000_000_000).contains(&ticks))
            {
                return Err(invalid(
                    "invalid scheduled tick, graph, capacity, queue policy, hazard, agent multipliers or service duration",
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
    let mut max_wait_ticks = o.max_wait_ticks;
    let mut max_queue_length = o.max_queue_length;
    let mut in_service = 0_u64;
    let mut queue = QueueState::new(o.population);
    let mut service_starts = Vec::new();
    let mut abandoned = Vec::new();
    let mut balks = Vec::new();
    let mut retries = Vec::new();
    let mut retry_policy = o.retry;
    let mut rng = DeterministicStream::from_run_seed(o.seed);
    let mut now = 0_u128;
    let horizon = (time.last().unwrap() * o.ticks_per_unit as f64).floor() as u128;
    if pause_tick.is_some_and(|tick| tick >= horizon) {
        return Err(invalid("pause_at_tick must be below the horizon"));
    }
    let mut metrics = ResourceAccounting::default();
    let mut completed = Vec::new();
    let mut trace = Vec::new();
    let mut requests = Vec::new();
    if let Some(token) = resume {
        let token = token
            .as_str()
            .ok_or_else(|| invalid("network checkpoint must be a string"))?;
        let saved: NetworkCheckpoint = serde_json::from_str(token)
            .map_err(|_| invalid("invalid network checkpoint format"))?;
        let max_slot_ticks = horizon * o.population as u128;
        let original_rng = rng.checkpoint();
        if saved.version != 1
            || saved.rng.seed != original_rng.seed
            || saved.rng.stream_id != original_rng.stream_id
            || saved.capacity > o.population as u64
            || saved.completed.len() > o.population - initial.len()
            || saved.completed.last().is_some_and(|tick| *tick > saved.now)
            || saved
                .pause_tick
                .map_or(saved.completed.last() != Some(&saved.now), |tick| {
                    tick < saved.now || tick >= horizon
                })
            || pause_tick.is_some_and(|tick| tick <= saved.pause_tick.unwrap_or(saved.now))
            || saved.completed.windows(2).any(|pair| pair[0] > pair[1])
            || saved
                .queue
                .retry_at
                .iter()
                .flatten()
                .any(|&tick| tick < saved.now)
            || [
                saved.metrics.occupied_slot_ticks,
                saved.metrics.over_capacity_slot_ticks,
                saved.metrics.configured_slot_ticks,
                saved.metrics.admitted_wait_ticks,
                saved.metrics.abandoned_wait_ticks,
            ]
            .into_iter()
            .any(|total| total > max_slot_ticks * 256)
            || saved
                .service
                .iter()
                .chain(saved.service_override.iter())
                .any(|&duration| !(1..=1_000_000_000_000).contains(&duration))
            || Some(&saved.configuration) != configuration.as_ref()
            || saved.now > horizon
            || pause.is_some_and(|count| count <= saved.completed.len() as u64)
            || saved.states.len() != o.population
            || saved.states.iter().any(|&state| state > 4)
            || [
                saved.queue.waiting_since.len(),
                saved.queue.deadlines.len(),
                saved.queue.retry_at.len(),
                saved.queue.retry_counts.len(),
                saved.priorities.len(),
                saved.service.len(),
                saved.innovation_weights.len(),
                saved.imitation_weights.len(),
            ]
            .into_iter()
            .any(|len| len != o.population)
            || saved.metrics.last_capacity_change > saved.now
            || saved.metrics.last_resource_change > saved.now
            || saved
                .service_starts
                .iter()
                .any(|entry| entry.agent >= o.population)
            || saved
                .queue
                .waiting_since
                .iter()
                .flatten()
                .any(|&tick| tick > saved.now)
            || saved
                .queue
                .deadlines
                .iter()
                .zip(&saved.queue.waiting_since)
                .any(|(deadline, arrival)| {
                    deadline.is_some_and(|end| {
                        end < saved.now || arrival.is_none_or(|start| end < start)
                    })
                })
        {
            return Err(invalid(
                "network checkpoint is incompatible with this simulation",
            ));
        }
        for &tick in &saved.applied_ticks {
            let Some((expected, replacement, _)) = changes.next() else {
                return Err(invalid("invalid checkpoint schedule cursor"));
            };
            if tick != expected || tick > saved.now {
                return Err(invalid("invalid checkpoint schedule cursor"));
            }
            if let Some(replacement) = replacement {
                graph = replacement;
            }
        }
        for (&agent, &state) in agents.iter().zip(&saved.states) {
            context.attach(agent, state);
        }
        let mut pending = std::collections::HashSet::new();
        for entry in &saved.service_starts {
            if saved.states[entry.agent] == 1 && saved.queue.waiting_since[entry.agent].is_none() {
                if entry.scheduled_completion_tick < saved.now || !pending.insert(entry.agent) {
                    return Err(invalid("invalid checkpoint pending completion"));
                }
                entry.schedule_completion(&mut context, agents[entry.agent]);
            }
        }
        if pending.len() as u64 != saved.in_service {
            return Err(invalid("invalid checkpoint service occupancy"));
        }
        rng = DeterministicStream::restore(&saved.rng)
            .map_err(|_| invalid("invalid checkpoint RNG state"))?;
        now = saved.now;
        applied_ticks = saved.applied_ticks;
        service_override = saved.service_override;
        capacity = saved.capacity;
        max_wait_ticks = saved.max_wait_ticks;
        max_queue_length = saved.max_queue_length;
        in_service = saved.in_service;
        queue = saved.queue;
        service_starts = saved.service_starts;
        abandoned = saved.abandoned;
        balks = saved.balks;
        retries = saved.retries;
        retry_policy = saved.retry_policy;
        metrics = saved.metrics;
        completed = saved.completed;
        trace = saved.trace;
        requests = saved.requests;
        innovation = saved.innovation;
        imitation = saved.imitation;
        innovation_weights = saved.innovation_weights;
        imitation_weights = saved.imitation_weights;
        priorities = saved.priorities;
        service = saved.service;
        resource = queue.rebuild_waiters(&agents, &priorities);
    }
    loop {
        let before_draw = pause_tick.map(|_| rng.checkpoint());
        let completion = context
            .scheduler
            .peek_next()
            .map(|e| e.at.ticks())
            .filter(|t| *t <= horizon);
        let boundary = changes
            .peek()
            .map(|(tick, _, _)| *tick)
            .filter(|t| *t <= horizon);
        let expiry = queue
            .deadlines
            .iter()
            .flatten()
            .copied()
            .min()
            .filter(|t| *t <= horizon);
        let wake = queue
            .retry_at
            .iter()
            .flatten()
            .copied()
            .min()
            .filter(|t| *t <= horizon);
        let agent_rates = |innovation: f64, imitation: f64| -> Vec<f64> {
            (0..o.population)
                .map(|i| {
                    if context.get::<u8>(agents[i]) != Some(&0) {
                        0.
                    } else {
                        let degree = graph.incoming[i].len();
                        let multiplier = imitation_weights[i];
                        let rate = imitation * multiplier;
                        let exposed_rate = |fraction| {
                            if rate.is_finite() {
                                rate * fraction
                            } else {
                                (imitation * fraction) * multiplier
                            }
                        };
                        innovation * innovation_weights[i]
                            + if degree == 0 {
                                0.
                            } else if graph.weighted {
                                let exposed: f64 = graph.incoming[i]
                                    .iter()
                                    .filter(|(j, _)| context.get::<u8>(agents[*j]) == Some(&2))
                                    .map(|(_, weight)| weight)
                                    .sum();
                                exposed_rate(exposed / graph.totals[i])
                            } else {
                                let exposed = graph.incoming[i]
                                    .iter()
                                    .filter(|(j, _)| context.get::<u8>(agents[*j]) == Some(&2))
                                    .count() as f64;
                                // Preserve seeded arithmetic for existing finite unforced paths.
                                let product = rate * exposed;
                                if o.hazard_forcing.is_none() && product.is_finite() {
                                    product / degree as f64
                                } else {
                                    exposed_rate(exposed / degree as f64)
                                }
                            }
                    }
                })
                .collect()
        };
        let mut rates = Vec::new();
        if o.hazard_forcing.is_none() {
            rates = agent_rates(innovation, imitation);
        }
        let mut total: f64 = rates.iter().sum();
        if !total.is_finite() {
            return Err(invalid("network hazard exceeded finite domain"));
        }
        let attempt = if let Some(forcing) = &o.hazard_forcing {
            let scale = o.ticks_per_unit as f64;
            let start = now as f64 / scale;
            let stop = [completion, boundary, expiry, wake]
                .into_iter()
                .flatten()
                .min()
                .unwrap_or(horizon);
            let end = stop as f64 / scale;
            // Normalize each basis before summing. Large weights can have a
            // finite effective sum once multiplied by a small coefficient.
            let scaled_basis = |rates: Vec<f64>| {
                let scale = rates.iter().copied().fold(0., f64::max);
                let sum: f64 = if scale == 0. {
                    0.
                } else {
                    rates.iter().map(|rate| rate / scale).sum()
                };
                (scale, sum)
            };
            let (innovation_scale, innovation_sum) = scaled_basis(agent_rates(1., 0.));
            let (imitation_scale, imitation_sum) = scaled_basis(agent_rates(0., 1.));
            if now >= stop || (innovation_scale == 0. && imitation_scale == 0.) {
                None
            } else {
                let budget: f64 = rng.sample(Exp::new(1.).expect("unit rate is positive"));
                let event_time = forcing
                    .attempt(start, end, budget, |t| {
                        let (p, q) = forcing.coefficients(t, innovation, imitation);
                        (p * innovation_scale) * innovation_sum
                            + (q * imitation_scale) * imitation_sum
                    })
                    .map_err(invalid)?;
                if let Some(t) = event_time {
                    let (p, q) = forcing.coefficients(t, innovation, imitation);
                    rates = agent_rates(p, q);
                    total = rates.iter().sum();
                    if !total.is_finite() || total <= 0. {
                        return Err(invalid(
                            "forced selection hazard outside finite positive domain",
                        ));
                    }
                }
                event_time.map(|t| ((t * scale).ceil() as u128).max(now + 1).min(horizon))
            }
        } else if total > 0. {
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
        if let Some(cutoff) = pause_tick {
            let next = [completion, boundary, expiry, wake, attempt]
                .into_iter()
                .flatten()
                .min();
            if next.is_none_or(|tick| tick > cutoff) {
                rng = DeterministicStream::restore(&before_draw.unwrap())
                    .map_err(|_| invalid("could not restore pending network hazard draw"))?;
                break;
            }
        }
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
            metrics.record_over_capacity(now, in_service, capacity);
            in_service -= 1;
        } else if let Some(tick) = expiry.filter(|t| {
            boundary.is_none_or(|b| *t <= b)
                && attempt.is_none_or(|a| *t <= a)
                && wake.is_none_or(|w| *t <= w)
        }) {
            now = tick;
            for id in queue
                .rebuild_deadlines(&agents)
                .expire_reneged(SimTime::from_ticks(now))
            {
                if queue.deadlines[id.index as usize].take().is_some() {
                    let index = id.index as usize;
                    queue.schedule_retry(retry_policy, index, now);
                    context.attach(
                        id,
                        if queue.retry_at[index].is_some() {
                            4_u8
                        } else {
                            3_u8
                        },
                    );
                    abandoned.push(json!({"tick":now,"agent":id.index}));
                    let queued_at = queue.waiting_since[index]
                        .take()
                        .ok_or_else(|| invalid("expired agent has no queue entry"))?;
                    metrics.abandoned_wait_ticks += now - queued_at;
                }
            }
            resource = queue.rebuild_waiters(&agents, &priorities);
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
            } else if let Some(values) = &change.agent_service_ticks {
                service.clone_from(values);
                service_override = None;
            }
            retry_policy = change.retry.unwrap_or(retry_policy);
            if let Some(new_capacity) = change.capacity {
                metrics.configured_slot_ticks +=
                    (now - metrics.last_capacity_change) * u128::from(capacity);
                metrics.last_capacity_change = now;
                metrics.record_over_capacity(now, in_service, capacity);
                capacity = new_capacity;
            }
            max_wait_ticks = change.max_wait_ticks.unwrap_or(max_wait_ticks);
            max_queue_length = change.max_queue_length.unwrap_or(max_queue_length);
            innovation = change.p.unwrap_or(innovation);
            imitation = change.q.unwrap_or(imitation);
            if let Some(values) = &change.innovation_multipliers {
                innovation_weights.clone_from(values);
            }
            if let Some(values) = &change.imitation_multipliers {
                imitation_weights.clone_from(values);
            }
            if let Some(values) = &change.agent_priorities {
                priorities.clone_from(values);
                resource = queue.rebuild_waiters(&agents, &priorities);
            }
            applied_ticks.push(tick);
        } else if let Some(tick) = wake.filter(|t| attempt.is_none_or(|a| *t <= a)) {
            now = tick;
            for (index, at) in queue.retry_at.iter_mut().enumerate() {
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
                && max_queue_length.is_some_and(|limit| resource.queue_length() >= limit)
            {
                queue.schedule_retry(retry_policy, index, now);
                context.attach(
                    id,
                    if queue.retry_at[index].is_some() {
                        4_u8
                    } else {
                        3_u8
                    },
                );
                balks.push(json!({"tick":now,"agent":id.index}));
                continue;
            }
            context.attach(id, 1_u8);
            queue.waiting_since[index] = Some(now);
            resource.request(id, priorities[index], SimTime::from_ticks(now));
            if let Some(wait) = max_wait_ticks.filter(|_| in_service >= capacity) {
                queue.deadlines[index] = Some(now + u128::from(wait));
            }
        } else {
            break;
        }
        while in_service < capacity {
            let Some(next) = resource.release() else {
                break;
            };
            let index = next.index as usize;
            let queued_at = queue.waiting_since[index]
                .take()
                .ok_or_else(|| invalid("service admission has no queue entry"))?;
            queue.deadlines[index] = None;
            let duration = service_override.unwrap_or(service[index]);
            let completion_tick = now + u128::from(duration);
            metrics.occupied_slot_ticks += completion_tick.min(horizon) - now;
            metrics.admitted_wait_ticks += now - queued_at;
            let admission = ServiceStart {
                agent: index,
                tick: now,
                request_tick: queued_at,
                wait_ticks: now - queued_at,
                service_ticks: duration,
                scheduled_completion_tick: completion_tick,
            };
            admission.schedule_completion(&mut context, next);
            service_starts.push(admission);
            metrics.record_over_capacity(now, in_service, capacity);
            in_service += 1;
        }
        metrics.peak_queue_length = metrics.peak_queue_length.max(resource.queue_length());
        metrics.peak_in_service = metrics.peak_in_service.max(in_service);
        if pause == Some(completed.len() as u64) {
            break;
        }
    }
    if pause_tick.is_some() || pause == Some(completed.len() as u64) {
        let checkpoint = NetworkCheckpoint {
            version: 1,
            pause_tick,
            configuration: configuration.unwrap(),
            states: agents
                .iter()
                .map(|agent| *context.get::<u8>(*agent).unwrap())
                .collect(),
            rng: rng.checkpoint(),
            now,
            applied_ticks,
            service_override,
            capacity,
            max_wait_ticks,
            max_queue_length,
            in_service,
            queue,
            service_starts,
            abandoned,
            balks,
            retries,
            retry_policy,
            metrics,
            completed,
            trace,
            requests,
            innovation,
            imitation,
            innovation_weights,
            imitation_weights,
            priorities,
            service,
        };
        let token = serde_json::to_string(&checkpoint)
            .map_err(|_| invalid("network checkpoint serialization failed"))?;
        return Ok(KernelResponse {
            schema_version: KERNEL_SCHEMA_VERSION.into(),
            operation: op,
            model_key: r.model_key.clone(),
            error: None,
            result: Some(json!({"status":"paused","checkpoint":token})),
            metadata: json!({"runtime":"rust_native","engine":"kairos_network"}),
        });
    }
    metrics.configured_slot_ticks +=
        (horizon - metrics.last_capacity_change) * u128::from(capacity);
    let pending_wait_ticks: u128 = queue
        .waiting_since
        .iter()
        .flatten()
        .map(|tick| horizon - tick)
        .sum();
    metrics.record_over_capacity(horizon, in_service, capacity);
    let idle_slot_ticks = (metrics.configured_slot_ticks + metrics.over_capacity_slot_ticks)
        .checked_sub(metrics.occupied_slot_ticks)
        .ok_or_else(|| invalid("inconsistent network resource accounting"))?;
    let resource_metrics = json!({
        "horizon_ticks":horizon,
        "peak_queue_length":metrics.peak_queue_length,
        "peak_in_service":metrics.peak_in_service,
        "occupied_slot_ticks":metrics.occupied_slot_ticks,
        "configured_slot_ticks":metrics.configured_slot_ticks,
        "over_capacity_slot_ticks":metrics.over_capacity_slot_ticks,
        "idle_slot_ticks":idle_slot_ticks,
        "admitted_wait_ticks":metrics.admitted_wait_ticks,
        "abandoned_wait_ticks":metrics.abandoned_wait_ticks,
        "pending_wait_ticks":pending_wait_ticks
    });
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
            json!({"values":values,"counts":counts,"agents":states,"requests":requests,"completions":trace,"service_starts":service_starts,"resource_metrics":resource_metrics,"abandonments":abandoned,"balks":balks,"retry_eligibility":retries,"metadata":{"engine":"kairos_network","kairos_revision":crate::simulation::REV,"seed":o.seed,"population":o.population,"capacity":o.capacity,"current_capacity":capacity,"current_p":innovation,"current_q":imitation,"in_service":in_service,"service_ticks":o.service_ticks,"ticks_per_unit":o.ticks_per_unit,"queue_length":queue_length,"available":available,"agent_priorities":priorities,"max_wait_ticks":o.max_wait_ticks,"current_max_wait_ticks":max_wait_ticks,"max_queue_length":o.max_queue_length,"current_max_queue_length":max_queue_length,"innovation_multipliers":innovation_weights,"imitation_multipliers":imitation_weights,"agent_service_ticks":service,"directed":o.directed,"weighted":graph.weighted,"applied_schedule_ticks":applied_ticks,"service_override_ticks":service_override,"hazard":if graph.weighted {"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted incoming weight/total incoming weight), zero imitation without incoming edges"} else {"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted neighbors/degree), zero imitation at degree zero"},"resource_semantics":"Nonpreemptive priority, FIFO ties; adoption occurs at service completion; pending agents do not transmit","kairos_dispatched_events":requests.len() + trace.len(),"rng_checkpoint":rng.checkpoint(),"checkpoint_scope":"RNG provenance only; use the network completion checkpoint for simulation resume"}}),
        ),
        metadata: json!({"runtime":"rust_native","engine":"kairos_network"}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuilt_deadlines_keep_original_expiry_and_arrival_order() {
        let mut context = ABMContext::new(42);
        let agents = [
            context.spawn_agent(),
            context.spawn_agent(),
            context.spawn_agent(),
        ];
        let mut queue = QueueState::new(agents.len());
        queue.waiting_since = vec![Some(3), Some(1), Some(4)];
        queue.deadlines = vec![Some(5), Some(5), None];
        let mut deadlines = queue.rebuild_deadlines(&agents);
        assert!(deadlines.expire_reneged(SimTime::from_ticks(4)).is_empty());
        assert_eq!(
            deadlines.expire_reneged(SimTime::from_ticks(5)),
            [agents[1], agents[0]]
        );
    }

    #[test]
    fn rebuilt_waiters_retain_priority_and_fifo_order() {
        let mut context = ABMContext::new(42);
        let agents = [
            context.spawn_agent(),
            context.spawn_agent(),
            context.spawn_agent(),
        ];
        let mut queue = QueueState::new(agents.len());
        queue.waiting_since = vec![Some(3), Some(1), Some(5)];
        let mut resource = queue.rebuild_waiters(&agents, &[0, 0, 1]);
        for index in [2, 1, 0] {
            assert_eq!(resource.release(), Some(agents[index]));
        }
        assert_eq!(resource.release(), None);
    }

    #[test]
    fn admission_order_reconstructs_tied_completion_order() {
        let mut context = ABMContext::new(42);
        let agents = [context.spawn_agent(), context.spawn_agent()];
        let admissions = [
            ServiceStart {
                agent: 1,
                tick: 1,
                request_tick: 1,
                wait_ticks: 0,
                service_ticks: 4,
                scheduled_completion_tick: 5,
            },
            ServiceStart {
                agent: 0,
                tick: 3,
                request_tick: 3,
                wait_ticks: 0,
                service_ticks: 2,
                scheduled_completion_tick: 5,
            },
        ];
        for admission in &admissions {
            admission.schedule_completion(&mut context, agents[admission.agent]);
        }
        for admission in &admissions {
            let StepOutcome::Dispatched(event) = context.step() else {
                panic!("expected pending completion");
            };
            assert_eq!(event.entity, Some(agents[admission.agent]));
            assert_eq!(event.at.ticks(), admission.scheduled_completion_tick);
        }
    }
}
