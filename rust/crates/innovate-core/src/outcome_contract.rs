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

/// Validate the paired Bass payload shared by JSON and Arrow consumers.
///
/// Requires at least two uniquely named strategies with provenance, matching
/// time units/perspectives, identical ordered draw IDs and one shared strictly
/// increasing time grid. Each draw retains finite, named `p`, `q` and `m`
/// parameters with matching units, plus aligned finite nonnegative adoption.
///
/// # Errors
/// Returns the first violated structural/value contract as a static explanation.
/// A successful validation does not establish causal comparability, economically
/// appropriate units or scientific calibration of the supplied trajectories.
pub fn validate_paired_outcomes(payload: &Value) -> Result<(), &'static str> {
    let strategies = payload["strategies"]
        .as_array()
        .ok_or("invalid paired outcome strategies")?;
    // Serde structs also accept positional arrays, but this contract uses named fields.
    if strategies.iter().any(|strategy| {
        !strategy.is_object()
            || strategy["draws"].as_array().is_some_and(|draws| {
                draws.iter().any(|draw| {
                    !draw.is_object()
                        || draw["parameters"]
                            .as_array()
                            .is_some_and(|parameters| parameters.iter().any(|p| !p.is_object()))
                })
            })
    }) {
        return Err("paired strategies, draws and parameters must be objects");
    }
    let strategies: Vec<Strategy> = Vec::deserialize(&payload["strategies"])
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
            if draw.adoption.iter().any(|&value| value < 0.) {
                return Err("paired Bass adoption values must be nonnegative");
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
    Ok(())
}
