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
    hazard_forcing: Option<HazardForcing>,

    #[serde(default)]
    schedule: Vec<Change>,
}

#[derive(Clone, Copy, Deserialize)]
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
    let mut reneging = RenegingQueue::new();
    let mut deadlines = vec![None; o.population];
    let mut service_starts = Vec::new();
    let mut abandoned = Vec::new();
    let mut balks = Vec::new();
    let mut retries = Vec::new();
    let mut retry_at = vec![None; o.population];
    let mut retry_counts = vec![0; o.population];
    let mut retry_policy = o.retry;
    let next_retry = |policy: Option<Retry>, index: usize, now: u128, counts: &mut [u8]| {
        let policy = policy?;
        if counts[index] >= policy.max_retries {
            return None;
        }
        counts[index] += 1;
        Some(now + u128::from(policy.delay_ticks))
    };
    let rebuild_waiters = |waiting_since: &[Option<u128>], priorities: &[i32]| {
        let mut waiting: Vec<_> = waiting_since
            .iter()
            .enumerate()
            .filter_map(|(index, tick)| tick.map(|tick| (tick, index)))
            .collect();
        waiting.sort_unstable();
        let mut resource = PriorityResource::new("adoption_waiters", 0);
        for (tick, index) in waiting {
            resource.request(agents[index], priorities[index], SimTime::from_ticks(tick));
        }
        resource
    };
    let mut waiting_since = vec![None; o.population];
    let mut rng = DeterministicStream::from_run_seed(o.seed);
    let mut now = 0_u128;
    let horizon = (time.last().unwrap() * o.ticks_per_unit as f64).floor() as u128;
    let mut occupied_slot_ticks = 0;
    let mut over_capacity_slot_ticks = 0;
    let mut last_resource_change = 0;
    let mut record_over_capacity = |tick, in_service: u64, capacity| {
        over_capacity_slot_ticks +=
            (tick - last_resource_change) * u128::from(in_service.saturating_sub(capacity));
        last_resource_change = tick;
    };
    let mut peak_queue_length = 0;
    let mut peak_in_service = 0;
    let mut configured_slot_ticks = 0;
    let mut last_capacity_change = 0;
    let mut admitted_wait_ticks = 0;
    let mut abandoned_wait_ticks = 0;
    let mut completed = Vec::new();
    let mut trace = Vec::new();
    let mut requests = Vec::new();
    loop {
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
            record_over_capacity(now, in_service, capacity);
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
                    retry_at[index] = next_retry(retry_policy, index, now, &mut retry_counts);
                    context.attach(
                        id,
                        if retry_at[index].is_some() {
                            4_u8
                        } else {
                            3_u8
                        },
                    );
                    abandoned.push(json!({"tick":now,"agent":id.index}));
                    let queued_at = waiting_since[index]
                        .take()
                        .ok_or_else(|| invalid("expired agent has no queue entry"))?;
                    abandoned_wait_ticks += now - queued_at;
                }
            }
            resource = rebuild_waiters(&waiting_since, &priorities);
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
                configured_slot_ticks += (now - last_capacity_change) * u128::from(capacity);
                last_capacity_change = now;
                record_over_capacity(now, in_service, capacity);
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
                resource = rebuild_waiters(&waiting_since, &priorities);
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
                && max_queue_length.is_some_and(|limit| resource.queue_length() >= limit)
            {
                retry_at[index] = next_retry(retry_policy, index, now, &mut retry_counts);
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
            if let Some(wait) = max_wait_ticks.filter(|_| in_service >= capacity) {
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
            let queued_at = waiting_since[index]
                .take()
                .ok_or_else(|| invalid("service admission has no queue entry"))?;
            deadlines[index] = None;
            let duration = service_override.unwrap_or(service[index]);
            let completion_tick = now + u128::from(duration);
            occupied_slot_ticks += completion_tick.min(horizon) - now;
            admitted_wait_ticks += now - queued_at;
            service_starts.push(json!({"agent":next.index,"tick":now,
                "request_tick":queued_at,"wait_ticks":now-queued_at,
                "service_ticks":duration,"scheduled_completion_tick":completion_tick}));
            context.schedule_behaviour(next, 2, SimTime::from_ticks(completion_tick));
            record_over_capacity(now, in_service, capacity);
            in_service += 1;
        }
        peak_queue_length = peak_queue_length.max(resource.queue_length());
        peak_in_service = peak_in_service.max(in_service);
    }
    configured_slot_ticks += (horizon - last_capacity_change) * u128::from(capacity);
    let pending_wait_ticks: u128 = waiting_since
        .iter()
        .flatten()
        .map(|tick| horizon - tick)
        .sum();
    record_over_capacity(horizon, in_service, capacity);
    let resource_metrics = json!({
        "horizon_ticks":horizon,
        "peak_queue_length":peak_queue_length,
        "peak_in_service":peak_in_service,
        "occupied_slot_ticks":occupied_slot_ticks,
        "configured_slot_ticks":configured_slot_ticks,
        "over_capacity_slot_ticks":over_capacity_slot_ticks,
        "idle_slot_ticks":configured_slot_ticks + over_capacity_slot_ticks - occupied_slot_ticks,
        "admitted_wait_ticks":admitted_wait_ticks,
        "abandoned_wait_ticks":abandoned_wait_ticks,
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
            json!({"values":values,"counts":counts,"agents":states,"requests":requests,"completions":trace,"service_starts":service_starts,"resource_metrics":resource_metrics,"abandonments":abandoned,"balks":balks,"retry_eligibility":retries,"metadata":{"engine":"kairos_network","kairos_revision":crate::simulation::REV,"seed":o.seed,"population":o.population,"capacity":o.capacity,"current_capacity":capacity,"current_p":innovation,"current_q":imitation,"in_service":in_service,"service_ticks":o.service_ticks,"ticks_per_unit":o.ticks_per_unit,"queue_length":queue_length,"available":available,"agent_priorities":priorities,"max_wait_ticks":o.max_wait_ticks,"current_max_wait_ticks":max_wait_ticks,"max_queue_length":o.max_queue_length,"current_max_queue_length":max_queue_length,"innovation_multipliers":innovation_weights,"imitation_multipliers":imitation_weights,"agent_service_ticks":service,"directed":o.directed,"weighted":graph.weighted,"applied_schedule_ticks":applied_ticks,"service_override_ticks":service_override,"hazard":if graph.weighted {"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted incoming weight/total incoming weight), zero imitation without incoming edges"} else {"p*innovation_multiplier[i] + q*imitation_multiplier[i]*(adopted neighbors/degree), zero imitation at degree zero"},"resource_semantics":"Nonpreemptive priority, FIFO ties; adoption occurs at service completion; pending agents do not transmit","kairos_dispatched_events":requests.len() + trace.len(),"rng_checkpoint":rng.checkpoint()}}),
        ),
        metadata: json!({"runtime":"rust_native","engine":"kairos_network"}),
    })
}
