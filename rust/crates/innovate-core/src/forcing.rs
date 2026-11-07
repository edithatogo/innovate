//! Piecewise-constant additive rate forcing for the native Bass simulation.
use serde::Deserialize;
use serde_json::Value;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Covariate {
    time: Vec<f64>,
    values: Vec<f64>,
    beta_p: f64,
    beta_q: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    time: f64,
    delta_p: f64,
    delta_q: f64,
}
pub(crate) fn schedule(
    covariate: Option<&Value>,
    events: Option<&Value>,
    ticks: u64,
    p: f64,
    q: f64,
) -> Result<Vec<(u128, f64, f64)>, String> {
    let cov: Option<Covariate> = covariate
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|e| e.to_string())?;
    let events: Vec<Event> = events
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let valid_time = |x: f64| x.is_finite() && x >= 0. && x * ticks as f64 <= u64::MAX as f64 / 2.;
    if let Some(c) = &cov
        && (c.time.is_empty()
            || c.time.len() != c.values.len()
            || c.time[0] != 0.
            || c.time.iter().any(|t| !valid_time(*t))
            || c.time.windows(2).any(|t| t[0] >= t[1])
            || c.values.iter().any(|x| !x.is_finite())
            || !c.beta_p.is_finite()
            || !c.beta_q.is_finite())
    {
        return Err("covariate needs aligned finite values, ordered times starting at zero, and finite coefficients".into());
    }
    if events
        .iter()
        .any(|e| !valid_time(e.time) || !e.delta_p.is_finite() || !e.delta_q.is_finite())
        || events.windows(2).any(|e| e[0].time >= e[1].time)
    {
        return Err(
            "events must have finite deltas and strictly increasing nonnegative times".into(),
        );
    }
    let mut times = vec![0.];
    if let Some(c) = &cov {
        times.extend(&c.time);
    }
    times.extend(events.iter().map(|e| e.time));
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut result = Vec::new();
    for t in times {
        let x = cov
            .as_ref()
            .map_or(0., |c| c.values[c.time.partition_point(|s| *s <= t) - 1]);
        let a = p
            + events
                .iter()
                .filter(|e| e.time <= t)
                .map(|e| e.delta_p)
                .sum::<f64>()
            + cov.as_ref().map_or(0., |c| c.beta_p * x);
        let b = q
            + events
                .iter()
                .filter(|e| e.time <= t)
                .map(|e| e.delta_q)
                .sum::<f64>()
            + cov.as_ref().map_or(0., |c| c.beta_q * x);
        if !a.is_finite() || !b.is_finite() || a < 0. || b < 0. {
            return Err(
                "forced innovation/imitation rates must stay finite and nonnegative".into(),
            );
        }
        let tick = (t * ticks as f64).ceil() as u128;
        if result
            .last()
            .is_some_and(|(previous, _, _)| *previous == tick)
        {
            if tick != 0 {
                return Err("forcing change times collide at the selected tick resolution".into());
            }
            result.pop();
        }
        result.push((tick, a, b));
    }
    Ok(result)
}
