//! Validate paired Bass adoption outputs before downstream economic evaluation.
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Strategy {
    scenario_id: String,
    time_unit: String,
    perspective: Option<String>,
    provenance: String,
    draws: Vec<Draw>,
}

#[derive(Deserialize)]
struct Draw {
    draw_id: u64,
    parameters: Vec<Parameter>,
    time: Vec<f64>,
    adoption: Vec<f64>,
}

#[derive(Deserialize)]
struct Parameter {
    name: String,
    unit: String,
    value: f64,
}

pub(crate) fn validate(payload: &Value) -> Result<Value, &'static str> {
    let strategies: Vec<Strategy> = serde_json::from_value(payload["strategies"].clone())
        .map_err(|_| "invalid paired outcome strategies")?;
    let reference = strategies
        .first()
        .filter(|_| strategies.len() >= 2)
        .ok_or("paired outcomes require at least two strategies")?;
    let mut identities = BTreeSet::new();
    let mut reference_grid = None;
    let mut reference_units = None;
    for strategy in &strategies {
        if strategy.scenario_id.trim().is_empty()
            || !identities.insert(&strategy.scenario_id)
            || strategy.provenance.trim().is_empty()
            || strategy.time_unit.trim().is_empty()
        {
            return Err("strategies require unique identities, provenance and time units");
        }
        if strategy.time_unit != reference.time_unit
            || strategy.perspective != reference.perspective
            || strategy
                .perspective
                .as_ref()
                .is_some_and(|value| value.trim().is_empty())
        {
            return Err("strategies require matching time units and perspectives");
        }
        if strategy.draws.is_empty()
            || !strategy
                .draws
                .iter()
                .map(|draw| draw.draw_id)
                .eq(reference.draws.iter().map(|draw| draw.draw_id))
        {
            return Err("strategies require matching ordered draw IDs");
        }
        let mut draw_ids = BTreeSet::new();
        for draw in &strategy.draws {
            if !draw_ids.insert(draw.draw_id)
                || draw.time.is_empty()
                || draw.time.len() != draw.adoption.len()
                || draw
                    .time
                    .iter()
                    .chain(&draw.adoption)
                    .any(|x| !x.is_finite())
                || draw.time.windows(2).any(|pair| pair[0] >= pair[1])
                || *reference_grid.get_or_insert(&draw.time) != &draw.time
            {
                return Err("draws require unique IDs and aligned trajectories on one time grid");
            }
            let mut units = BTreeMap::new();
            for parameter in &draw.parameters {
                if parameter.unit.trim().is_empty()
                    || !parameter.value.is_finite()
                    || units.insert(&parameter.name, &parameter.unit).is_some()
                {
                    return Err("invalid or duplicate draw parameter");
                }
            }
            if !["m", "p", "q"]
                .into_iter()
                .eq(units.keys().map(|name| name.as_str()))
                || reference_units.get_or_insert(units.clone()) != &units
            {
                return Err("Bass draws require matching p, q and m parameter units");
            }
        }
    }
    Ok(payload.clone())
}
