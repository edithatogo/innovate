//! Finite-population Bass pure-birth process. Kairos owns time, dispatch and RNG.
use crate::validation::{numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use kairo_ecs_core::Scheduler;
use kairo_ecs_rng::{DeterministicStream, rand_distr::Exp};
use kairo_ecs_types::{EventKind, ScheduleRequest, SimTime, StepOutcome};
use serde_json::{Value, json};
pub(crate) const REV: &str = "d56863505d938feb069af88a737eb7668113d14b";
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
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
    if options
        .keys()
        .any(|k| !["engine", "population", "seed", "ticks_per_unit"].contains(&k.as_str()))
        || options.get("engine").and_then(Value::as_str) != Some("kairos_des")
    {
        return Err(invalid("unknown simulation engine or option"));
    }
    let seed = options
        .get("seed")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("seed must be an unsigned integer"))?;
    let population = options
        .get("population")
        .and_then(Value::as_u64)
        .filter(|n| *n > 0 && *n <= 100000)
        .ok_or_else(|| invalid("population must be in 1..=100000"))?;
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
    let horizon = (*time.last().unwrap() * ticks as f64).floor() as u128;
    let mut scheduler = Scheduler::new();
    let mut rng = DeterministicStream::from_run_seed(seed);
    let mut events = Vec::new();
    let mut now = 0_u128;
    for adopted in 0..population {
        let rate = (p + q * adopted as f64 / population as f64) * (population - adopted) as f64;
        if !rate.is_finite() {
            return Err(invalid("simulation hazard exceeded finite domain"));
        }
        let distribution =
            Exp::new(rate).map_err(|_| invalid("simulation hazard must be positive"))?;
        let delay: f64 = rng.sample(distribution);
        let delta = (delay * ticks as f64).ceil().max(1.);
        if !delta.is_finite() || delta > horizon.saturating_sub(now) as f64 {
            break;
        }
        now += delta as u128;
        scheduler.schedule(ScheduleRequest {
            at: SimTime::from_ticks(now),
            priority: 0,
            entity: None,
            kind: EventKind::Custom(1),
        });
        match scheduler.step() {
            StepOutcome::Dispatched(event) => events.push(event.at.ticks()),
            _ => {
                return Err(invalid(
                    "Kairos dispatch did not return the scheduled adoption",
                ));
            }
        }
    }
    let counts: Vec<_> = time
        .iter()
        .map(|t| events.partition_point(|e| *e <= (*t * ticks as f64).floor() as u128) as u64)
        .collect();
    let values: Vec<_> = counts
        .iter()
        .map(|n| m * (*n as f64 / population as f64))
        .collect();
    let result = json!({"values":values,"counts":counts,"shape":[time.len()],"dtype":"float64","metadata":{"engine":"kairos_des","kairos_revision":REV,"seed":seed,"population":population,"ticks_per_unit":ticks,"quantity":"adopted_level","process":"finite_population_bass_pure_birth","hazard":"(p + q*n/N)*(N-n)","time_quantization":"waiting times rounded up to at least one tick","initial_adopters":0,"rng_checkpoint":rng.checkpoint(),"checkpoint_scope":"RNG provenance only; simulation resume is not implemented"}});
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        result: Some(result),
        error: None,
        metadata: json!({"runtime":"rust_native","engine":"kairos_des"}),
    })
}
