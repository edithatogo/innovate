//! Multi-generation substitution. Historical ODE and successive-generation semantics are explicit.
use crate::events::ParameterEvents;
use crate::mechanistic::History;
use crate::numerics::least_squares;
use crate::validation::{numeric_array, numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use serde_json::{Map, Value, json};

type Error = KernelBindingError;
type Trajectory = (Vec<Vec<f64>>, Vec<Vec<f64>>);
#[derive(Clone)]
struct Model {
    n: usize,
    variant: String,
    launches: Vec<f64>,
    origin: f64,
    initial: Vec<f64>,
    covariates: Vec<String>,
    history: Option<History>,
    events: ParameterEvents,
}
fn invalid(op: KernelOperation, message: impl Into<String>) -> Error {
    Error::invalid_request(op, message)
}

fn options(request: &KernelRequest, inputs: &Map<String, Value>) -> Result<Model, Error> {
    let op = request.operation;
    let p = request
        .payload
        .as_object()
        .ok_or_else(|| invalid(op, "payload must be an object"))?;
    let kw = p
        .get("constructor_kwargs")
        .or_else(|| p.get("model_kwargs"))
        .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")));
    let empty = Map::new();
    let kw = match kw {
        Some(v) => v
            .as_object()
            .ok_or_else(|| invalid(op, "constructor_kwargs must be an object"))?,
        None => &empty,
    };
    for key in kw.keys() {
        if ![
            "n_generations",
            "variant",
            "launch_times",
            "initial_time",
            "initial_values",
            "covariates",
            "covariate_history",
            "parameter_events",
        ]
        .contains(&key.as_str())
        {
            return Err(Error::unsupported_native_operation(
                op,
                format!("Norton-Bass option '{key}' is not implemented"),
            ));
        }
    }
    if p.get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != "norton_bass")
    {
        return Err(invalid(op, "request and state model_key differ"));
    }
    for value in [
        p.get("predict_kwargs"),
        p.get("state").and_then(|s| s.get("predict_kwargs")),
        p.get("fitter_options"),
    ]
    .into_iter()
    .flatten()
    {
        if !value.as_object().is_some_and(Map::is_empty) {
            return Err(Error::unsupported_native_operation(
                op,
                "custom prediction/fitter options are unavailable",
            ));
        }
    }
    let n = if let Some(n) = kw.get("n_generations") {
        n.as_u64()
            .ok_or_else(|| invalid(op, "n_generations must be a positive integer"))?
            as usize
    } else if let Some(parameters) = p
        .get("state")
        .and_then(|s| s.get("parameters"))
        .and_then(Value::as_object)
    {
        parameters
            .keys()
            .filter(|k| {
                k.strip_prefix('p')
                    .is_some_and(|i| i.parse::<usize>().is_ok())
            })
            .count()
            .max(1)
    } else {
        inputs
            .get("observed")
            .or_else(|| inputs.get("y"))
            .and_then(Value::as_array)
            .and_then(|r| r.first())
            .and_then(Value::as_array)
            .map_or(1, Vec::len)
    };
    if n == 0 || n > 32 {
        return Err(invalid(
            op,
            "n_generations must be in 1..=32 for this native implementation",
        ));
    }
    let variant = kw
        .get("variant")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| invalid(op, "variant must be a string"))
        })
        .transpose()?
        .unwrap_or("legacy_cannibalization")
        .to_string();
    if !["legacy_cannibalization", "successive_generations"].contains(&variant.as_str()) {
        return Err(invalid(op, "unknown Norton-Bass variant"));
    }
    let launches = kw
        .get("launch_times")
        .map(|v| numeric_array(v, "launch_times", op))
        .transpose()?
        .unwrap_or(vec![0.0; n]);
    let origin = kw
        .get("initial_time")
        .map(|v| {
            v.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| invalid(op, "initial_time must be finite"))
        })
        .transpose()?
        .unwrap_or(0.0);
    let initial = kw
        .get("initial_values")
        .map(|v| numeric_array(v, "initial_values", op))
        .transpose()?
        .unwrap_or(vec![0.0; n]);
    if launches.len() != n
        || launches.iter().any(|x| !x.is_finite() || *x < 0.0)
        || launches.windows(2).any(|p| p[0] > p[1])
    {
        return Err(invalid(
            op,
            "launch_times must be finite, nonnegative, ordered and match n_generations",
        ));
    }
    if initial.len() != n || initial.iter().any(|x| !x.is_finite() || *x < 0.0) || origin < 0.0 {
        return Err(invalid(op, "invalid initial time/state"));
    }
    if variant == "legacy_cannibalization" && launches.iter().any(|x| *x != 0.0) {
        return Err(invalid(
            op,
            "launch_times apply to successive_generations; use initial_time for the historical ODE",
        ));
    }
    if variant == "successive_generations" && (origin != 0.0 || initial.iter().any(|x| *x != 0.0)) {
        return Err(invalid(
            op,
            "initial state applies to legacy_cannibalization only",
        ));
    }
    let covariates: Vec<String> = kw
        .get("covariates")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| invalid(op, "covariates must be an array of names"))
        })
        .transpose()?
        .unwrap_or_default();
    let distinct: std::collections::HashSet<_> = covariates.iter().collect();
    if covariates.len() > 32
        || distinct.len() != covariates.len()
        || covariates.iter().any(|n| n.trim().is_empty())
    {
        return Err(invalid(
            op,
            "covariates must contain at most 32 distinct nonempty names",
        ));
    }
    let history = kw
        .get("covariate_history")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| invalid(op, "invalid covariate_history"))
        })
        .transpose()?;
    let base_names: Vec<_> = (1..=n)
        .flat_map(|index| ['p', 'q', 'm'].map(|name| format!("{name}{index}")))
        .collect();
    let mut names = base_names.clone();
    for covariate in &covariates {
        names.extend(
            base_names
                .iter()
                .map(|name| format!("beta_{name}_{covariate}")),
        );
    }
    let events = ParameterEvents::parse(kw.get("parameter_events"), &names, 3 * n)
        .map_err(|error| invalid(op, error))?;
    Ok(Model {
        n,
        variant,
        launches,
        origin,
        initial,
        covariates,
        history,
        events,
    })
}

fn parameters(value: &Value, model: &Model, op: KernelOperation) -> Result<Vec<f64>, Error> {
    let p = value
        .as_object()
        .ok_or_else(|| invalid(op, "parameters must be an object"))?;
    if p.len() != 3 * model.n * (1 + model.covariates.len()) {
        return Err(invalid(
            op,
            "parameters must contain exactly p_i, q_i, m_i and declared beta coefficients",
        ));
    }
    let mut result = Vec::new();
    for i in 1..=model.n {
        for key in ['p', 'q', 'm'] {
            let value = required_f64(p, &format!("{key}{i}"), op)?;
            if !value.is_finite() || value < 0.0 || (key == 'm' && value == 0.0) {
                return Err(invalid(
                    op,
                    "p_i/q_i must be nonnegative and m_i positive, all finite",
                ));
            }
            result.push(value);
        }
    }
    for name in &model.covariates {
        for i in 1..=model.n {
            for key in ['p', 'q', 'm'] {
                result.push(required_f64(p, &format!("beta_{key}{i}_{name}"), op)?);
            }
        }
    }
    Ok(result)
}
fn parameter_value(p: &[f64], model: &Model) -> Value {
    let mut out = Map::new();
    for (i, p) in p[..3 * model.n].as_chunks::<3>().0.iter().enumerate() {
        for (key, value) in ['p', 'q', 'm'].into_iter().zip(p) {
            out.insert(format!("{key}{}", i + 1), json!(value));
        }
    }
    let mut betas = p[3 * model.n..].iter();
    for name in &model.covariates {
        for i in 1..=model.n {
            for key in ['p', 'q', 'm'] {
                out.insert(
                    format!("beta_{key}{i}_{name}"),
                    json!(betas.next().unwrap()),
                );
            }
        }
    }
    Value::Object(out)
}

// Bass fraction and its right derivative. Zero innovation and zero initial state imply no adoption.
fn fraction(t: f64, p: f64, q: f64) -> (f64, f64) {
    if t < 0.0 || p == 0.0 {
        return (0.0, 0.0);
    }
    let decay = (-(p + q) * t).exp();
    let f = p * -(-(p + q) * t).exp_m1() / (p + q * decay);
    (f, (p + q * f) * (1.0 - f))
}
impl Model {
    fn threshold_diagnostics(&self, parameters: &[f64]) -> Value {
        let generations: Vec<_> = (0..self.n)
            .map(|index| {
                let innovation = parameters[3 * index];
                let imitation = parameters[3 * index + 1];
                let (peak, takeoff, saturation) = if innovation > 0. && imitation > 0. {
                    let scale = innovation.max(imitation);
                    let normalized_sum = innovation / scale + imitation / scale;
                    let log_ratio = imitation.ln() - innovation.ln();
                    let duration = |numerator: f64| (numerator / normalized_sum) / scale;
                    (
                        duration(log_ratio),
                        duration(log_ratio - 9_f64.ln()).max(0.),
                        duration(log_ratio + 9_f64.ln()),
                    )
                } else {
                    (0., 0., 0.)
                };
                json!({
                    "generation": index + 1,
                    "reference_time": if self.variant == "successive_generations" {
                        self.launches[index]
                    } else {
                        self.origin
                    },
                    "half_life": peak,
                    "takeoff": takeoff,
                    "saturation": saturation,
                    "status": if innovation == 0. || imitation == 0. {
                        "degenerate_rates"
                    } else if [peak, takeoff, saturation].iter().any(|value| !value.is_finite()) {
                        "outside_finite_domain"
                    } else {
                        "available"
                    }
                })
            })
            .collect();
        json!({
            "method": "historical_bass_threshold_heuristic",
            "scope": "unforced_base_parameters",
            "time_basis": "elapsed_from_reference_time",
            "half_life_semantics": "historical label for unconstrained Bass inflection offset",
            "generations": generations
        })
    }

    fn trajectory_rate_peaks(&self, time: &[f64], rates: &[Vec<f64>]) -> Value {
        let generations: Vec<_> = (0..self.n)
            .map(|index| {
                let reference = if self.variant == "successive_generations" {
                    self.launches[index]
                } else {
                    self.origin
                };
                let peak = time
                    .iter()
                    .copied()
                    .zip(rates)
                    .skip_while(|(t, _)| *t < reference)
                    .map(|(t, row)| (t, row[index]))
                    .reduce(|best, sample| if sample.1 > best.1 { sample } else { best });
                match peak {
                    Some((t, rate)) => json!({
                        "generation": index + 1,
                        "status": "sampled",
                        "peak_time": t,
                        "peak_rate": rate
                    }),
                    None => json!({"generation": index + 1, "status": "no_active_samples"}),
                }
            })
            .collect();
        json!({
            "method": "maximum_signed_generation_rate_on_requested_grid",
            "event_side": "right_hand_excluding_jump_impulses",
            "tie_break": "earliest_sample",
            "window": [time.first(), time.last()],
            "generations": generations
        })
    }

    fn kwargs(&self) -> Value {
        let mut kwargs = json!({"n_generations":self.n,"variant":self.variant,"launch_times":self.launches,"initial_time":self.origin,"initial_values":self.initial});
        if let Some(history) = &self.history {
            kwargs["covariates"] = json!(self.covariates);
            kwargs["covariate_history"] = json!(history);
        }
        if !self.events.is_empty() {
            kwargs["parameter_events"] = self.events.value();
        }
        kwargs
    }
    fn derivative(&self, t: f64, y: &[f64], p: &[f64]) -> Vec<f64> {
        self.derivative_in_regime(t, y, p, self.events.regime(t))
    }
    fn derivative_in_regime(&self, t: f64, y: &[f64], p: &[f64], regime: usize) -> Vec<f64> {
        let p = self.events.parameters(p, regime);
        let controls: Vec<_> = self
            .history
            .iter()
            .flat_map(|history| self.covariates.iter().map(move |name| history.at(name, t)))
            .collect();
        (0..self.n)
            .map(|i| {
                let mut coefficients = [p[3 * i], p[3 * i + 1], p[3 * i + 2]];
                for (control, betas) in controls
                    .iter()
                    .zip(p[3 * self.n..].chunks_exact(3 * self.n))
                {
                    for (coefficient, beta) in coefficients.iter_mut().zip(&betas[3 * i..3 * i + 3])
                    {
                        *coefficient += beta * control;
                    }
                }
                let [innovation, imitation, market] = coefficients;
                if coefficients.iter().any(|v| !v.is_finite()) {
                    return f64::NAN;
                }
                if market <= 0. {
                    return 0.;
                }
                (innovation + imitation * y[i] / market) * (market - y[i..].iter().sum::<f64>())
            })
            .collect()
    }
    fn successive_coefficients(
        &self,
        t: f64,
        active: &[f64],
        generation: usize,
    ) -> Result<[f64; 3], String> {
        let offset = 3 * generation;
        let mut coefficients = [active[offset], active[offset + 1], active[offset + 2]];
        if let Some(history) = &self.history {
            for (index, name) in self.covariates.iter().enumerate() {
                let start = 3 * self.n * (index + 1) + offset;
                let beta = &active[start..start + 3];
                for (coefficient, beta) in coefficients.iter_mut().zip(beta) {
                    *coefficient += beta * history.at(name, t);
                }
            }
        }
        if coefficients
            .iter()
            .any(|value| !value.is_finite() || *value < 0.)
            || coefficients[2] == 0.
        {
            return Err("successive-generation effective p/q must be finite nonnegative and m finite positive".into());
        }
        Ok(coefficients)
    }

    fn successive_market_rate(&self, t: f64, active: &[f64], generation: usize) -> f64 {
        self.history.as_ref().map_or(0., |history| {
            self.covariates
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    let beta = active[3 * self.n * (index + 1) + 3 * generation + 2];
                    history.scaled_slope(name, t, beta)
                })
                .sum()
        })
    }

    fn intrinsic_milestones(
        &self,
        times: &[f64],
        states: &[Vec<f64>],
        derivative: impl Fn(f64, &[f64], usize) -> Vec<f64>,
        generation: usize,
    ) -> Result<Value, String> {
        let mut row =
            json!({"generation": generation + 1, "reference_time": self.launches[generation]});
        for (name, target) in [("takeoff", 0.1_f64), ("median", 0.5), ("saturation", 0.9)] {
            let level = -(-target).ln_1p();
            let crossing = states.iter().position(|state| state[0] >= level);
            let Some(index) = crossing else {
                row[name] = Value::Null;
                continue;
            };
            // A positive milestone cannot be met by the zero state at launch.
            let origin = times[index - 1];
            let initial = &states[index - 1];
            let mut lower = origin;
            let mut upper = times[index];
            for _ in 0..48 {
                let midpoint = lower.midpoint(upper);
                if midpoint == lower || midpoint == upper {
                    break;
                }
                let state = self
                    .events
                    .integrate(&[midpoint], origin, initial, &derivative)?;
                if state[0][0] >= level {
                    upper = midpoint;
                } else {
                    lower = midpoint;
                }
            }
            row[name] = json!(upper);
        }
        Ok(row)
    }

    fn predict_successive_forced(
        &self,
        time: &[f64],
        parameters: &[f64],
        mut milestones: Option<&mut Vec<Value>>,
    ) -> Result<Trajectory, String> {
        let end = *time.last().ok_or("time must not be empty")?;
        let mut grid = time.to_vec();
        if let Some(history) = &self.history {
            grid.extend(
                history
                    .time
                    .iter()
                    .copied()
                    .filter(|t| *t >= 0. && *t <= end),
            );
        }
        grid.extend(self.launches.iter().copied().filter(|t| *t <= end));
        grid.extend(self.events.times().filter(|t| *t <= end));
        grid.sort_by(f64::total_cmp);
        grid.dedup_by(|left, right| *left == *right);
        let mut log_survival = vec![vec![0.; self.n]; time.len()];
        for (generation, &launch) in self.launches.iter().enumerate() {
            let active_times: Vec<_> = grid.iter().copied().filter(|t| *t >= launch).collect();
            for &t in &active_times {
                let regime = self.events.regime(t);
                let left = self.events.boundary_left_regime(t).filter(|_| t > launch);
                for side in std::iter::once(regime).chain(left) {
                    let active = self.events.parameters(parameters, side);
                    self.successive_coefficients(t, &active, generation)?;
                }
            }
            // Log survival keeps the launched Bass fraction in [0, 1].
            let derivative = |t, state: &[f64], regime| {
                let active = self.events.parameters(parameters, regime);
                let rate = self
                    .successive_coefficients(t, &active, generation)
                    .map(|[p, q, _]| p + q * -(-state[0]).exp_m1())
                    .unwrap_or(f64::NAN);
                vec![rate]
            };
            let states = if active_times.is_empty() {
                Vec::new()
            } else {
                self.events
                    .integrate(&active_times, launch, &[0.], derivative)?
            };
            if let Some(rows) = &mut milestones {
                rows.push(self.intrinsic_milestones(
                    &active_times,
                    &states,
                    derivative,
                    generation,
                )?);
            }
            for (index, &t) in time.iter().enumerate() {
                if t >= launch {
                    log_survival[index][generation] =
                        states[active_times.partition_point(|point| *point < t)][0];
                }
            }
        }
        let mut rows = Vec::with_capacity(time.len());
        let mut rates = Vec::with_capacity(time.len());
        for (&t, state) in time.iter().zip(&log_survival) {
            let active = self.events.parameters(parameters, self.events.regime(t));
            let mut quantities = Vec::with_capacity(self.n);
            for (generation, (&launch, &value)) in self.launches.iter().zip(state).enumerate() {
                if t < launch {
                    quantities.push((0., 0., 0., 0.));
                    continue;
                }
                let [p, q, market] = self.successive_coefficients(t, &active, generation)?;
                // Capacity derivatives belong to the right-hand report, not the fraction ODE.
                let market_rate = self.successive_market_rate(t, &active, generation);
                if !market_rate.is_finite() {
                    return Err(
                        "successive-generation capacity slope exceeded the finite numerical domain"
                            .into(),
                    );
                }
                let fraction = -(-value).exp_m1();
                let fraction_rate = (p + q * fraction) * (-value).exp();
                quantities.push((fraction, fraction_rate, market, market_rate));
            }
            let mut levels = Vec::with_capacity(self.n);
            let mut derivatives = Vec::with_capacity(self.n);
            let mut inherited = 0.;
            let mut inherited_rate = 0.;
            for (generation, &(fraction, fraction_rate, market, market_rate)) in
                quantities.iter().enumerate()
            {
                let level = fraction * (market + inherited);
                let rate = fraction_rate * (market + inherited)
                    + fraction * (market_rate + inherited_rate);
                let (next, next_rate, _, _) = quantities
                    .get(generation + 1)
                    .copied()
                    .unwrap_or((0., 0., 0., 0.));
                levels.push(level * (1. - next));
                derivatives.push(rate * (1. - next) - level * next_rate);
                inherited = level;
                inherited_rate = rate;
            }
            rows.push(levels);
            rates.push(derivatives);
        }
        if rows
            .iter()
            .chain(&rates)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(
                "successive-generation prediction exceeded the finite numerical domain".into(),
            );
        }
        Ok((rows, rates))
    }

    fn predict(&self, time: &[f64], p: &[f64]) -> Result<Trajectory, String> {
        if self.variant == "legacy_cannibalization" {
            let rows: Vec<Vec<f64>> = if self.history.is_none()
                && self.events.is_empty()
                && self.n == 1
                && self.initial[0] == 0.0
            {
                if time.iter().any(|t| *t < self.origin) {
                    return Err("time precedes initial_time".into());
                }
                time.iter()
                    .map(|t| vec![p[2] * fraction(t - self.origin, p[0], p[1]).0])
                    .collect()
            } else {
                let mut grid = time.to_vec();
                if let Some(history) = &self.history {
                    let end = time.last().copied().unwrap_or(self.origin);
                    grid.extend(
                        history
                            .time
                            .iter()
                            .copied()
                            .filter(|t| *t > self.origin && *t < end),
                    );
                    grid.sort_by(f64::total_cmp);
                    grid.dedup_by(|a, b| *a == *b);
                }
                let states =
                    self.events
                        .integrate(&grid, self.origin, &self.initial, |t, y, regime| {
                            self.derivative_in_regime(t, y, p, regime)
                        })?;
                time.iter()
                    .map(|t| states[grid.partition_point(|v| v < t)].clone())
                    .collect()
            };
            let rates: Vec<Vec<f64>> = time
                .iter()
                .zip(&rows)
                .map(|(&t, y)| self.derivative(t, y, p))
                .collect();
            if rows.iter().chain(&rates).flatten().any(|x| !x.is_finite()) {
                return Err("prediction exceeded the finite numerical domain".into());
            }
            return Ok((rows, rates));
        }
        if self.history.is_some() || !self.events.is_empty() {
            return self.predict_successive_forced(time, p, None);
        }
        let mut rows = Vec::new();
        let mut rates = Vec::new();
        for &t in time {
            let fractions: Vec<_> = (0..self.n)
                .map(|i| fraction(t - self.launches[i], p[3 * i], p[3 * i + 1]))
                .collect();
            let mut levels = Vec::new();
            let mut derivatives = Vec::new();
            let mut inherited = 0.0;
            let mut inherited_rate = 0.0;
            for i in 0..self.n {
                let (f, df) = fractions[i];
                let market = p[3 * i + 2] + inherited;
                let level = f * market;
                let rate = df * market + f * inherited_rate;
                let (next, next_rate) = fractions.get(i + 1).copied().unwrap_or((0.0, 0.0));
                levels.push(level * (1.0 - next));
                derivatives.push(rate * (1.0 - next) - level * next_rate);
                inherited = level;
                inherited_rate = rate;
            }
            rows.push(levels);
            rates.push(derivatives);
        }
        if rows.iter().chain(&rates).flatten().any(|x| !x.is_finite()) {
            return Err("prediction exceeded the finite numerical domain".into());
        }
        Ok((rows, rates))
    }
    fn prediction_value(&self, time: &[f64], rows: &[Vec<f64>], rates: &[Vec<f64>]) -> Value {
        let warnings = if rows.iter().flatten().any(|x| *x < 0.0) {
            vec![
                "Historical cannibalization equations produced negative levels; inspect the model/market assumptions",
            ]
        } else {
            vec![]
        };
        json!({"columns":(1..=self.n).map(|i|format!("series_{i}")).collect::<Vec<_>>(),"rows":rows,"rates":rates,
            "metadata":{"shape":[time.len(),self.n],"variant":self.variant,"quantity":"generation_level",
            "rate_semantics":"signed derivative of generation level; not gross incidence", "launch_derivative":"right_hand", "parameter_event_semantics":if self.variant=="successive_generations" {"continuous launched fractions; capacity overrides can jump levels; right-hand rates exclude jump impulses"} else {"persistent fixed overrides; continuous historical ODE state; right-hand rates"}, "warnings":warnings}})
    }
}

pub(crate) fn observations(
    inputs: &Map<String, Value>,
    n: usize,
    length: usize,
    op: KernelOperation,
) -> Result<Vec<Vec<f64>>, Error> {
    let value = inputs
        .get("observed")
        .or_else(|| inputs.get("y"))
        .ok_or_else(|| invalid(op, "observed is required"))?;
    let rows = value
        .as_array()
        .ok_or_else(|| invalid(op, "observed must be an array"))?;
    if rows.len() != length {
        return Err(invalid(op, "time and observed lengths differ"));
    }
    rows.iter()
        .map(|v| {
            let values = if n == 1 && v.is_number() {
                vec![v.as_f64().unwrap()]
            } else {
                numeric_array(v, "observed row", op)?
            };
            if values.len() != n || values.iter().any(|x| !x.is_finite()) {
                return Err(invalid(
                    op,
                    "observed rows must be finite and match n_generations",
                ));
            }
            Ok(values)
        })
        .collect()
}
pub(crate) fn diagnostics(
    time: &[f64],
    observed: &[Vec<f64>],
    predicted: &[Vec<f64>],
    name: &str,
    parameter_count: usize,
) -> Value {
    let n = observed[0].len();
    let y: Vec<_> = observed.iter().flatten().copied().collect();
    let p: Vec<_> = predicted.iter().flatten().copied().collect();
    let mut d = crate::kernel::fit_diagnostics_count(time, &y, &p, name, parameter_count + 1);
    let means: Vec<f64> = (0..n)
        .map(|i| {
            observed
                .iter()
                .map(|r| r[i] / observed.len() as f64)
                .sum::<f64>()
        })
        .collect();
    let total: f64 = observed
        .iter()
        .flat_map(|row| row.iter().zip(&means).map(|(y, m)| (y - m).powi(2)))
        .sum();
    let rss = d["metrics"]["RSS"].as_f64().unwrap();
    let r2 = if total > 0.0 && total.is_finite() {
        json!(1.0 - rss / total)
    } else {
        Value::Null
    };
    d["metrics"]["R_squared"] = r2.clone();
    d["metrics"]["R-squared"] = r2;
    d["observation_shape"] = json!([time.len(), n]);
    d["residual_order"] = json!("row_major");
    d["residual_analysis"] = json!({"support_level":"unavailable","reason":"multivariate temporal significance tests are not implemented"});
    d
}

pub(crate) fn execute(request: &KernelRequest) -> Result<KernelResponse, Error> {
    let op = request.operation;
    let payload = request
        .payload
        .as_object()
        .ok_or_else(|| invalid(op, "payload must be an object"))?;
    let inputs = object_section(payload, "inputs", op)?;
    let mut model = options(request, inputs)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], op)?;
    if time.iter().any(|t| !t.is_finite() || *t < 0.0) || time.windows(2).any(|t| t[1] <= t[0]) {
        return Err(invalid(
            op,
            "time must be finite, nonnegative and strictly increasing",
        ));
    }
    let forcing_origin = if model.variant == "successive_generations" {
        model.launches[0]
    } else {
        model.origin
    };
    model.history = History::merge(
        model.history.take(),
        &model.covariates,
        &time,
        inputs.get("covariates"),
        forcing_origin,
    )
    .map_err(|e| invalid(op, e))?;
    let mut optimizer = None;
    let mut identifiability = None;
    let p = if op == KernelOperation::FitModel {
        if time.len() < 4 {
            return Err(invalid(op, "fitting requires at least four time points"));
        }
        let observed = observations(inputs, model.n, time.len(), op)?;
        let y_scale = observed
            .iter()
            .flatten()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        let t_scale = time.last().copied().unwrap().max(1.0);
        if y_scale == 0.0 {
            return Err(invalid(
                op,
                "all-zero observations cannot identify diffusion parameters",
            ));
        }
        let fit_options = payload
            .get("fit_options")
            .map(|v| {
                v.as_object()
                    .ok_or_else(|| invalid(op, "fit_options must be an object"))
            })
            .transpose()?;
        if fit_options.is_some_and(|o| {
            o.keys()
                .any(|k| !["initial_parameters", "max_iterations"].contains(&k.as_str()))
        }) {
            return Err(Error::unsupported_native_operation(
                op,
                "unsupported fit option",
            ));
        }
        let max_iterations = fit_options
            .and_then(|o| o.get("max_iterations"))
            .map(|v| {
                v.as_u64()
                    .filter(|v| *v > 0 && *v <= 2000)
                    .ok_or_else(|| invalid(op, "max_iterations must be in 1..=2000"))
            })
            .transpose()?
            .unwrap_or(300) as usize;
        let transform = |p: Vec<f64>| {
            p.iter()
                .enumerate()
                .map(|(i, p)| {
                    if i >= 3 * model.n {
                        return *p;
                    }
                    (if i % 3 == 2 { p / y_scale } else { p * t_scale })
                        .max(1e-12)
                        .ln()
                })
                .collect::<Vec<_>>()
        };
        let untransform = |theta: &[f64]| {
            theta
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if i >= 3 * model.n {
                        return *t;
                    }
                    if i % 3 == 2 {
                        t.exp() * y_scale
                    } else {
                        t.exp() / t_scale
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut starts = Vec::new();
        if let Some(v) = fit_options.and_then(|o| o.get("initial_parameters")) {
            starts.push(transform(parameters(v, &model, op)?));
        }
        if model.history.is_some() && starts.is_empty() {
            return Err(invalid(
                op,
                "covariate fitting requires complete initial_parameters",
            ));
        }
        for p0 in if model.history.is_none() {
            &[0.1, 0.5, 1.0][..]
        } else {
            &[]
        } {
            for q0 in [1.0, 5.0, 10.0] {
                let mut start = Vec::new();
                for i in 0..model.n {
                    let max = observed
                        .iter()
                        .map(|r| r[i].abs())
                        .fold(0.0, f64::max)
                        .max(y_scale * 0.1);
                    start.extend([p0 / t_scale, q0 / t_scale, 1.2 * max]);
                }
                starts.push(transform(start));
            }
        }
        let target: Vec<_> = observed.iter().flatten().map(|y| y / y_scale).collect();
        let fit = least_squares(
            &target,
            &starts,
            |theta| {
                model
                    .predict(&time, &untransform(theta))
                    .map(|(rows, _)| rows.into_iter().flatten().map(|x| x / y_scale).collect())
            },
            max_iterations,
        )
        .map_err(|e| invalid(op, e))?;
        optimizer = Some(fit.report);
        identifiability = Some(fit.identifiability);
        untransform(&fit.parameters)
    } else {
        let value = payload
            .get("parameters")
            .or_else(|| payload.get("state").and_then(|s| s.get("parameters")))
            .ok_or_else(|| invalid(op, "parameters are required"))?;
        parameters(value, &model, op)?
    };
    let mut milestones = Vec::new();
    let forced = model.history.is_some() || !model.events.is_empty();
    let (predicted, rates) = if forced
        && model.variant == "successive_generations"
        && !matches!(
            op,
            KernelOperation::PredictModel | KernelOperation::SimulateModel
        ) {
        model.predict_successive_forced(&time, &p, Some(&mut milestones))
    } else {
        model.predict(&time, &p)
    }
    .map_err(|e| invalid(op, e))?;
    let predictions = model.prediction_value(&time, &predicted, &rates);
    let state = json!({"model_key":"norton_bass","model_name":"NortonBassModel","parameters":parameter_value(&p, &model),"constructor_kwargs":model.kwargs(),"predict_kwargs":{}});
    let result = match op {
        KernelOperation::PredictModel | KernelOperation::SimulateModel => {
            let mut predictions = predictions;
            if model.history.is_some() || !model.events.is_empty() {
                predictions["state"] = state;
            }
            predictions
        }
        _ => {
            let mut result = json!({"model_key":"norton_bass","model_name":"NortonBassModel","family":"substitution","parameters":parameter_value(&p, &model),"state":state,"predictions":predictions});
            result["threshold_diagnostics"] = model.threshold_diagnostics(&p);
            if forced {
                result["threshold_diagnostics"]["trajectory"] =
                    model.trajectory_rate_peaks(&time, &rates);
            }
            if !milestones.is_empty() {
                result["threshold_diagnostics"]["intrinsic_adoption_milestones"] = json!({
                    "quantity": "intrinsic_launched_fraction",
                    "time_basis": "absolute_model_time",
                    "evaluated_through": time.last(),
                    "fractions": {"takeoff": 0.1, "median": 0.5, "saturation": 0.9},
                    "generations": milestones
                });
            }
            if op != KernelOperation::SummarizeModel
                || inputs.contains_key("observed")
                || inputs.contains_key("y")
            {
                let observed = observations(inputs, model.n, time.len(), op)?;
                let mut d = diagnostics(&time, &observed, &predicted, "NortonBassModel", p.len());
                if !model.events.is_empty() {
                    d["parameter_events"] = json!({"count":model.events.len(),"parameters_estimated":false,"state_reset":false});
                }
                if let Some(report) = optimizer {
                    if report["converged"] != true {
                        d["warnings"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!("optimizer did not converge"));
                    }
                    d["optimizer"] = report;
                }
                if let Some(report) = identifiability {
                    if report["status"] != "local_full_rank" {
                        d["warnings"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!("parameters are locally rank deficient"));
                    }
                    d["identifiability"] = report;
                }
                result["diagnostics"] = d;
            }
            result
        }
    };
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: Some("norton_bass".into()),
        result: Some(result),
        error: None,
        metadata: json!({"model_key":"norton_bass","model_name":"NortonBassModel","family":"substitution","runtime":"rust_native"}),
    })
}
