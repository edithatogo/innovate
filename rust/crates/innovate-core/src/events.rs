//! Validated fixed parameter interventions and continuous-state ODE segmentation.
use crate::numerics::integrate;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Cow, collections::BTreeMap};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ParameterEvent {
    time: f64,
    parameters: BTreeMap<String, f64>,
}

#[derive(Clone, Default)]
pub(crate) struct ParameterEvents {
    events: Vec<ParameterEvent>,
    overrides: Vec<Vec<(usize, f64)>>,
}

impl ParameterEvents {
    pub(crate) fn parse(
        value: Option<&Value>,
        names: &[String],
        base_count: usize,
    ) -> Result<Self, String> {
        let events: Vec<ParameterEvent> = value
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()
            .map_err(|_| "parameter_events must contain times and named parameter overrides")?
            .unwrap_or_default();
        if events.len() > 32
            || events.iter().any(|event| {
                !event.time.is_finite() || event.time < 0. || event.parameters.is_empty()
            })
            || events.windows(2).any(|pair| pair[0].time >= pair[1].time)
        {
            return Err("parameter_events requires at most 32 nonempty changes at finite increasing nonnegative times".into());
        }
        let indices: BTreeMap<_, _> = names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect();
        let mut overrides = Vec::with_capacity(events.len());
        for event in &events {
            let mut entries = Vec::with_capacity(event.parameters.len());
            for (name, &value) in &event.parameters {
                let index = indices
                    .get(name.as_str())
                    .copied()
                    .ok_or_else(|| format!("unknown event parameter '{name}'"))?;
                let invalid_base =
                    index < base_count && (value < 0. || (name.starts_with('m') && value == 0.));
                if !value.is_finite() || invalid_base {
                    return Err("event base parameters must retain their physical domain and beta coefficients must be finite".into());
                }
                entries.push((index, value));
            }
            overrides.push(entries);
        }
        Ok(Self { events, overrides })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }

    pub(crate) fn value(&self) -> Value {
        serde_json::to_value(&self.events).expect("validated finite event parameters")
    }

    pub(crate) fn times(&self) -> impl Iterator<Item = f64> + '_ {
        self.events.iter().map(|event| event.time)
    }

    pub(crate) fn regime(&self, time: f64) -> usize {
        self.events.partition_point(|event| event.time <= time)
    }

    pub(crate) fn boundary_left_regime(&self, time: f64) -> Option<usize> {
        let active = self.regime(time);
        active
            .checked_sub(1)
            .filter(|&index| self.events[index].time == time)
    }

    pub(crate) fn parameters<'a>(&self, parameters: &'a [f64], regime: usize) -> Cow<'a, [f64]> {
        if regime == 0 {
            return Cow::Borrowed(parameters);
        }
        let mut active = parameters.to_vec();
        for overrides in &self.overrides[..regime] {
            for &(index, value) in overrides {
                active[index] = value;
            }
        }
        Cow::Owned(active)
    }

    pub(crate) fn integrate(
        &self,
        times: &[f64],
        origin: f64,
        initial: &[f64],
        derivative: impl Fn(f64, &[f64], usize) -> Vec<f64>,
    ) -> Result<Vec<Vec<f64>>, String> {
        if self.is_empty() {
            return integrate(times, origin, initial, |time, state| {
                derivative(time, state, 0)
            });
        }
        let end = *times.last().ok_or("time must not be empty")?;
        let mut grid = times.to_vec();
        grid.extend(self.times().filter(|time| *time > origin && *time <= end));
        grid.sort_by(f64::total_cmp);
        grid.dedup_by(|left, right| *left == *right);
        let mut states = Vec::with_capacity(grid.len());
        let mut start = origin;
        let mut current = initial.to_vec();
        let mut cursor = 0;
        let mut regime = self.regime(origin);
        while cursor < grid.len() {
            let boundary = self
                .events
                .get(regime)
                .map_or(end, |event| event.time.min(end));
            let stop = grid.partition_point(|time| *time <= boundary);
            let segment = integrate(&grid[cursor..stop], start, &current, |time, state| {
                derivative(time, state, regime)
            })?;
            current = segment
                .last()
                .ok_or("event interval has no boundary state")?
                .clone();
            states.extend(segment);
            cursor = stop;
            start = boundary;
            regime += 1;
        }
        Ok(times
            .iter()
            .map(|time| states[grid.partition_point(|point| point < time)].clone())
            .collect())
    }
}
