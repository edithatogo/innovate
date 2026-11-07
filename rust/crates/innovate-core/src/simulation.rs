//! Finite-population Bass pure-birth process. Kairos owns time, dispatch and RNG.
use crate::validation::{numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use kairo_ecs_abm::ABMContext;
use kairo_ecs_core::Scheduler;
use kairo_ecs_rng::{
    DeterministicStream,
    rand_distr::{Exp, Uniform},
};
use kairo_ecs_types::{EntityId, EventKind, ScheduleRequest, SimTime, StepOutcome};
use serde_json::{Value, json};
pub(crate) const REV: &str = "d56863505d938feb069af88a737eb7668113d14b";
struct AdoptionClock {
    rng: DeterministicStream,
    now: u128,
    forcing_index: usize,
}

impl AdoptionClock {
    fn next_adoption(
        &mut self,
        forcing: &[(u128, f64, f64)],
        adopted: u64,
        population: u64,
        ticks: u64,
        horizon: u128,
    ) -> Result<Option<u128>, &'static str> {
        loop {
            let (_, p, q) = forcing[self.forcing_index];
            let next = forcing.get(self.forcing_index + 1).map(|row| row.0);
            let rate = (p + q * adopted as f64 / population as f64) * (population - adopted) as f64;
            if !rate.is_finite() {
                return Err("simulation hazard exceeded finite domain");
            }
            let delta = if rate == 0. {
                f64::INFINITY
            } else {
                let delay: f64 = self
                    .rng
                    .sample(Exp::new(rate).map_err(|_| "invalid simulation hazard")?);
                (delay * ticks as f64).ceil().max(1.)
            };
            if let Some(boundary) = next.filter(|t| *t <= horizon)
                && delta >= boundary.saturating_sub(self.now) as f64
            {
                self.now = boundary;
                self.forcing_index += 1;
                continue;
            }
            if !delta.is_finite() || delta > horizon.saturating_sub(self.now) as f64 {
                return Ok(None);
            }
            self.now += delta as u128;
            return Ok(Some(self.now));
        }
    }
}

pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    if r.payload["simulation_options"]["engine"] == "kairos_network" {
        return crate::network::execute(r);
    }
    let op = r.operation;
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
    if op != KernelOperation::SimulateModel || r.model_key.as_deref() != Some("bass") {
        return Err(KernelBindingError::unsupported_native_operation(
            op,
            "Kairos execution currently supports Bass simulate_model only",
        ));
    }
    let payload = r
        .payload
        .as_object()
        .ok_or_else(|| invalid("payload must be an object"))?;
    let options = object_section(payload, "simulation_options", op)?;
    if options.keys().any(|k| {
        ![
            "engine",
            "population",
            "seed",
            "ticks_per_unit",
            "initial_adopters",
            "covariate",
            "events",
        ]
        .contains(&k.as_str())
    }) || !matches!(
        options.get("engine").and_then(Value::as_str),
        Some("kairos_des" | "kairos_abm")
    ) {
        return Err(invalid("unknown simulation engine or option"));
    }
    let engine = options["engine"].as_str().unwrap();
    let seed = options
        .get("seed")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("seed must be an unsigned integer"))?;
    let population = options
        .get("population")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0 && *n <= 100000)
        .ok_or_else(|| invalid("population must be in 1..=100000"))?;
    let initial = options
        .get("initial_adopters")
        .map(|v| {
            v.as_u64()
                .filter(|n| *n <= population)
                .ok_or_else(|| invalid("initial_adopters must be in 0..=population"))
        })
        .transpose()?
        .unwrap_or(0);
    let ticks = options
        .get("ticks_per_unit")
        .map(|v| {
            v.as_u64()
                .filter(|n| *n > 0 && *n <= 1000000000)
                .ok_or_else(|| invalid("ticks_per_unit must be in 1..=1000000000"))
        })
        .transpose()?
        .unwrap_or(1000000);
    let input = object_section(payload, "inputs", op)?;
    let time = numeric_array_from_aliases(input, &["time", "t"], op)?;
    if time
        .iter()
        .any(|t| !t.is_finite() || *t < 0. || *t * ticks as f64 > u64::MAX as f64 / 2.)
    {
        return Err(invalid(
            "simulation time exceeds the nonnegative finite tick domain",
        ));
    }
    let parameters = payload
        .get("state")
        .and_then(|s| s.get("parameters"))
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("parameters required"))?;
    if parameters.len() != 3 {
        return Err(invalid("Bass simulation requires exactly p, q and m"));
    }
    let p = required_f64(parameters, "p", op)?;
    let q = required_f64(parameters, "q", op)?;
    let m = required_f64(parameters, "m", op)?;
    let forcing =
        crate::forcing::schedule(options.get("covariate"), options.get("events"), ticks, p, q)
            .map_err(|e| invalid(&e))?;
    let horizon = (*time.last().unwrap() * ticks as f64).floor() as u128;
    let mut scheduler = Scheduler::new();
    let mut abm = (engine == "kairos_abm").then(|| ABMContext::new(seed));
    let agents: Vec<_> = if let Some(context) = abm.as_mut() {
        (0..population)
            .map(|_| {
                let id = context.spawn_agent();
                context.attach(id, false);
                id
            })
            .collect()
    } else {
        vec![]
    };
    let mut susceptible = agents.clone();
    let mut identity_rng = DeterministicStream::from_entity_purpose(
        seed,
        EntityId::new(0, 0),
        "innovate.adoption_identity",
    );
    if let Some(context) = abm.as_mut() {
        for _ in 0..initial {
            let agent = susceptible.swap_remove(identity_rng.sample(
                Uniform::new(0, susceptible.len()).map_err(|_| invalid("empty population"))?,
            ));
            context.attach(agent, true);
        }
    }
    let mut clock = AdoptionClock {
        rng: DeterministicStream::from_run_seed(seed),
        now: 0,
        forcing_index: 0,
    };
    let mut events = Vec::new();
    for adopted in initial..population {
        let Some(now) = clock
            .next_adoption(&forcing, adopted, population, ticks, horizon)
            .map_err(invalid)?
        else {
            break;
        };
        let outcome = if let Some(context) = abm.as_mut() {
            let agent = susceptible.swap_remove(
                identity_rng.sample(
                    Uniform::new(0, susceptible.len())
                        .map_err(|_| invalid("empty susceptible population"))?,
                ),
            );
            context.schedule_behaviour(agent, 1, SimTime::from_ticks(now));
            context.step()
        } else {
            scheduler.schedule(ScheduleRequest {
                at: SimTime::from_ticks(now),
                priority: 0,
                entity: None,
                kind: EventKind::Custom(1),
            });
            scheduler.step()
        };
        match outcome {
            StepOutcome::Dispatched(event) => {
                if let Some(context) = abm.as_mut() {
                    let id = event
                        .entity
                        .ok_or_else(|| invalid("Kairos lost agent identity"))?;
                    if context.get::<bool>(id) != Some(&false) {
                        return Err(invalid("agent adoption state is inconsistent"));
                    }
                    context.attach(id, true);
                }
                events.push(event.at.ticks());
            }
            _ => {
                return Err(invalid(
                    "Kairos dispatch did not return the scheduled adoption",
                ));
            }
        }
    }
    let counts: Vec<_> = time
        .iter()
        .map(|t| {
            events.partition_point(|e| *e <= (*t * ticks as f64).floor() as u128) as u64 + initial
        })
        .collect();
    let values: Vec<_> = counts
        .iter()
        .map(|n| m * (*n as f64 / population as f64))
        .collect();
    let mut result = json!({"values":values,"counts":counts,"shape":[time.len()],"dtype":"float64","metadata":{"engine":engine,"kairos_revision":REV,"seed":seed,"population":population,"ticks_per_unit":ticks,"quantity":"adopted_level","process":"finite_population_bass_pure_birth","hazard":"(p + q*n/N)*(N-n)","time_quantization":"waiting times rounded up to at least one tick","forcing":forcing,"forcing_semantics":"right-continuous additive covariate and event deltas; boundaries precede tied adoption events","initial_adopters":initial,"kairos_dispatched_events":events.len(),"rng_checkpoint":clock.rng.checkpoint(),"checkpoint_scope":"RNG provenance only; simulation resume is not implemented"}});
    if let Some(context) = abm.as_ref() {
        result["agents"]=json!(agents.iter().map(|id|json!({"index":id.index,"generation":id.generation,"adopted":context.get::<bool>(*id).copied().unwrap_or(false)})).collect::<Vec<_>>());
        result["metadata"]["agent_semantics"] = json!(
            "exchangeable homogeneous agents; identities sampled independently of event timing"
        );
    }
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        result: Some(result),
        error: None,
        metadata: json!({"runtime":"rust_native","engine":engine}),
    })
}
