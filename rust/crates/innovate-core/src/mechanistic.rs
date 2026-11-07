//! Bass/Gompertz ODE forcing with explicit origin, event and history conventions.
use crate::numerics::{integrate, least_squares};
use crate::validation::{numeric_array_from_aliases, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct History {
    pub(crate) time: Vec<f64>,
    values: BTreeMap<String, Vec<f64>>,
}
impl History {
    pub(crate) fn merge(
        mut history: Option<Self>,
        names: &[String],
        time: &[f64],
        values: Option<&Value>,
        origin: f64,
    ) -> Result<Option<Self>, String> {
        if let Some(existing) = &history {
            existing.validate(names)?;
        }
        if let Some(values) = values {
            let incoming = Self {
                time: time.to_vec(),
                values: serde_json::from_value(values.clone())
                    .map_err(|_| "covariates must map names to numeric arrays")?,
            };
            incoming.validate(names)?;
            if let Some(existing) = &mut history {
                let last = *existing.time.last().unwrap();
                for (i, &t) in time.iter().enumerate() {
                    if t <= last {
                        if names
                            .iter()
                            .any(|name| existing.at(name, t) != incoming.values[name][i])
                        {
                            return Err(
                                "covariate history revision requires an explicit new history/state"
                                    .into(),
                            );
                        }
                    } else {
                        existing.time.push(t);
                        for name in names {
                            existing
                                .values
                                .get_mut(name)
                                .unwrap()
                                .push(incoming.values[name][i]);
                        }
                    }
                }
                existing.validate(names)?;
            } else {
                history = Some(incoming);
            }
        }
        if names.is_empty() {
            if history.is_some() {
                return Err("forcing history requires declared covariates".into());
            }
        } else if history.as_ref().is_none_or(|h| h.time[0] > origin) {
            return Err("covariate history must cover the initial time; provide origin data or explicit history".into());
        }
        Ok(history)
    }

    fn validate(&self, names: &[String]) -> Result<(), String> {
        if self.time.is_empty()
            || self.time.iter().any(|v| !v.is_finite())
            || self
                .time
                .windows(2)
                .any(|w| w[0] >= w[1] || !(w[1] - w[0]).is_finite())
            || self.values.len() != names.len()
            || names.iter().any(|n| {
                self.values
                    .get(n)
                    .is_none_or(|v| v.len() != self.time.len() || v.iter().any(|v| !v.is_finite()))
            })
        {
            return Err("forcing history must have finite increasing times and exact aligned covariate names".into());
        }
        Ok(())
    }
    pub(crate) fn scaled_slope(&self, name: &str, t: f64, coefficient: f64) -> f64 {
        if coefficient == 0. {
            return 0.;
        }
        let index = self.time.partition_point(|value| *value <= t);
        if index == 0 || index == self.time.len() {
            return 0.;
        }
        let values = &self.values[name];
        let width = self.time[index] - self.time[index - 1];
        let difference = values[index] - values[index - 1];
        if difference == 0. {
            return 0.;
        }
        let slope = if difference.is_finite() {
            difference / width
        } else {
            values[index] / width - values[index - 1] / width
        };
        if slope.is_finite() && slope != 0. {
            return coefficient * slope;
        }
        // Reorder scaling when an intermediate slope overflows or underflows.
        let scaled_difference = if difference.is_finite() {
            coefficient * difference
        } else {
            coefficient * values[index] - coefficient * values[index - 1]
        };
        let result = scaled_difference / width;
        if result.is_finite() && scaled_difference != 0. {
            return result;
        }
        (coefficient / width) * difference
    }

    pub(crate) fn at(&self, name: &str, t: f64) -> f64 {
        let i = self.time.partition_point(|v| *v <= t);
        let v = &self.values[name];
        if i == 0 {
            v[0]
        } else if i == self.time.len() {
            v[i - 1]
        } else {
            let f = (t - self.time[i - 1]) / (self.time[i] - self.time[i - 1]);
            (1. - f) * v[i - 1] + f * v[i]
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Options {
    #[serde(default)]
    covariates: Vec<String>,
    #[serde(default)]
    t_event: Option<f64>,
    #[serde(default = "mode")]
    prediction_mode: String,
    #[serde(default)]
    initial_time: Option<f64>,
    #[serde(default)]
    initial_value: Option<f64>,
    #[serde(default)]
    covariate_history: Option<History>,
    #[serde(default = "layout")]
    coefficient_layout: String,
    #[serde(default = "domain")]
    effective_domain: String,
    #[serde(default = "projection")]
    projection: String,
}
fn mode() -> String {
    "ode_fixed".into()
}
fn layout() -> String {
    "corrected".into()
}
fn domain() -> String {
    "physical".into()
}
fn projection() -> String {
    "raw".into()
}
pub(crate) fn selected(r: &KernelRequest) -> bool {
    if !matches!(r.model_key.as_deref(), Some("bass" | "gompertz"))
        || r.payload.get("simulation_options").is_some()
    {
        return false;
    }
    r.payload
        .get("constructor_kwargs")
        .or_else(|| r.payload.get("model_kwargs"))
        .or_else(|| {
            r.payload
                .get("state")
                .and_then(|s| s.get("constructor_kwargs"))
        })
        .is_some_and(|k| {
            k.get("prediction_mode").is_some()
                || k.get("event_candidates").is_some()
                || k.get("t_event").is_some_and(|v| !v.is_null())
                || k.get("covariates")
                    .and_then(Value::as_array)
                    .is_some_and(|v| !v.is_empty())
        })
}
struct Parameter {
    name: String,
    positive: bool,
    capacity: bool,
    unused: bool,
}
struct Model {
    key: String,
    options: Options,
    spec: Vec<Parameter>,
    origin: f64,
    initial: f64,
    forcing: Option<History>,
}
impl Model {
    fn parse(&self, v: &Value) -> Result<Vec<f64>, String> {
        let p = v.as_object().ok_or("parameters must be an object")?;
        if p.len() != self.spec.len() {
            return Err("parameters must match base, event and declared beta names".into());
        }
        self.spec.iter().map(|s|{let v=required_f64(p,&s.name,KernelOperation::PredictModel).map_err(|e|e.to_string())?;if !v.is_finite() || (s.positive && v<0.) || (s.positive && s.capacity && v<=0.){Err("base coefficients must be nonnegative and capacities positive; beta coefficients finite signed values".into())}else{Ok(v)}}).collect()
    }
    fn value(&self, p: &[f64]) -> Value {
        Value::Object(
            self.spec
                .iter()
                .zip(p)
                .map(|(s, v)| (s.name.clone(), json!(v)))
                .collect(),
        )
    }
    fn post(&self, t: f64) -> bool {
        self.options.t_event.is_some_and(|event| t >= event)
    }
    fn effective(&self, t: f64, post: bool, p: &[f64]) -> Result<[f64; 3], String> {
        let start = if post { 3 } else { 0 };
        let mut a = [p[start], p[start + 1], p[start + 2]];
        let beta = if self.options.t_event.is_some()
            && (post || self.options.coefficient_layout == "corrected")
        {
            6
        } else {
            3
        };
        for (i, name) in self.options.covariates.iter().enumerate() {
            let x = self.forcing.as_ref().unwrap().at(name, t);
            for j in 0..3 {
                a[j] += p[beta + 3 * i + j] * x;
            }
        }
        let valid = a.iter().all(|v| v.is_finite())
            && (self.options.effective_domain == "historical"
                || if self.key == "bass" {
                    a[0] >= 0. && a[1] >= 0. && a[2] > 0.
                } else {
                    a[0] > 0. && a[2] >= 0.
                });
        if valid {
            Ok(a)
        } else {
            Err("effective ODE coefficients exceed the declared finite/physical domain".into())
        }
    }
    fn derivative(&self, t: f64, y: f64, post: bool, p: &[f64]) -> Result<f64, String> {
        let a = self.effective(t, post, p)?;
        let rate = if self.key == "bass" {
            if a[2] <= 0. {
                0.
            } else {
                (a[0] + a[1] * y / a[2]) * (a[2] - y)
            }
        } else if a[0] <= 0. || y <= 0. {
            0.
        } else {
            a[2] * y * (a[0].ln() - y.ln())
        };
        if rate.is_finite() {
            Ok(rate)
        } else {
            Err("ODE growth rate overflow".into())
        }
    }
    fn predict(&self, time: &[f64], p: &[f64]) -> Result<Value, String> {
        let end = *time.last().ok_or("prediction time required")?;
        let mut grid = vec![self.origin];
        grid.extend_from_slice(time);
        if let Some(h) = &self.forcing {
            grid.extend(
                h.time
                    .iter()
                    .filter(|t| **t >= self.origin && **t <= end)
                    .copied(),
            );
        }
        if let Some(event) = self
            .options
            .t_event
            .filter(|t| *t > self.origin && *t <= end)
        {
            grid.push(event);
        }
        grid.sort_by(f64::total_cmp);
        grid.dedup_by(|a, b| *a == *b);
        for t in &grid {
            self.effective(*t, self.post(*t), p)?;
        }
        let rows = if let Some(event) = self
            .options
            .t_event
            .filter(|t| *t > self.origin && *t <= end)
        {
            self.effective(event, false, p)?;
            let split = grid.partition_point(|t| *t <= event);
            let mut left = integrate(&grid[..split], self.origin, &[self.initial], |t, y| {
                vec![self.derivative(t, y[0], false, p).unwrap_or(f64::NAN)]
            })?;
            let current = left.last().unwrap()[0];
            left.extend(integrate(&grid[split..], event, &[current], |t, y| {
                vec![self.derivative(t, y[0], true, p).unwrap_or(f64::NAN)]
            })?);
            left
        } else {
            let post = self.post(self.origin);
            integrate(&grid, self.origin, &[self.initial], |t, y| {
                vec![self.derivative(t, y[0], post, p).unwrap_or(f64::NAN)]
            })?
        };
        if rows.iter().flatten().any(|v| !v.is_finite()) {
            return Err("ODE trajectory overflow".into());
        }
        let raw: Vec<_> = time
            .iter()
            .map(|t| rows[grid.partition_point(|v| v < t)][0])
            .collect();
        let mut values = raw.clone();
        if self.options.projection == "cumulative_max" {
            if self.options.prediction_mode == "ode_historical_batch" {
                let mut max = f64::NEG_INFINITY;
                for v in &mut values {
                    max = max.max(*v);
                    *v = max;
                }
            } else {
                let mut max = f64::NEG_INFINITY;
                let maxima: Vec<_> = rows
                    .iter()
                    .map(|row| {
                        max = max.max(row[0]);
                        max
                    })
                    .collect();
                values = time
                    .iter()
                    .map(|t| maxima[grid.partition_point(|v| v < t)])
                    .collect();
            }
        }
        let rates: Vec<Vec<f64>> = time
            .iter()
            .zip(&raw)
            .map(|(t, y)| self.derivative(*t, *y, self.post(*t), p).map(|v| vec![v]))
            .collect::<Result<_, _>>()?;
        Ok(
            json!({"values":values,"raw_values":raw,"rates":rates,"shape":[time.len()],"dtype":"float64","metadata":{"prediction_mode":self.options.prediction_mode,"initial_time":self.origin,"initial_value":self.initial,"projection":self.options.projection,"coefficient_layout":self.options.coefficient_layout,"effective_domain":self.options.effective_domain,"rate_semantics":"underlying signed ODE RHS at raw state; right-hand event rate, not derivative of projected report","forcing_semantics":"linear interpolation; endpoint hold; saved historical forcing cannot be silently revised","simulation_semantics":"deterministic ODE path; stochastic draws unavailable"}}),
        )
    }
}
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = r.operation;
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
    let key = r.model_key.as_deref().unwrap();
    let base = if key == "bass" {
        ["p", "q", "m"]
    } else {
        ["a", "b", "c"]
    };
    let p = r
        .payload
        .as_object()
        .ok_or_else(|| invalid("payload must be an object"))?;
    if p.keys().any(|k| {
        ![
            "inputs",
            "constructor_kwargs",
            "model_kwargs",
            "state",
            "parameters",
            "fit_options",
            "predict_kwargs",
            "fitter_options",
        ]
        .contains(&k.as_str())
    }) {
        return Err(invalid("unknown mechanistic payload option"));
    }
    for v in [
        p.get("predict_kwargs"),
        p.get("fitter_options"),
        p.get("state").and_then(|s| s.get("predict_kwargs")),
    ]
    .into_iter()
    .flatten()
    {
        if !v.as_object().is_some_and(Map::is_empty) {
            return Err(invalid("custom prediction/fitter options unavailable"));
        }
    }
    if p.get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != key)
    {
        return Err(invalid("state model_key mismatch"));
    }
    let empty = Map::new();
    let input = p
        .get("inputs")
        .map(|v| {
            v.as_object()
                .ok_or_else(|| invalid("inputs must be an object"))
        })
        .transpose()?
        .unwrap_or(&empty);
    let time = if op == KernelOperation::SummarizeModel
        && !input.contains_key("time")
        && !input.contains_key("t")
    {
        vec![]
    } else {
        numeric_array_from_aliases(input, &["time", "t"], op)?
    };
    if time.iter().any(|t| !t.is_finite() || *t < 0.) || time.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(
            "time must be finite nonnegative and strictly increasing",
        ));
    }
    let options = p
        .get("constructor_kwargs")
        .or_else(|| p.get("model_kwargs"))
        .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")));
    if let Some(options) = options
        .and_then(Value::as_object)
        .filter(|options| options.contains_key("event_candidates"))
    {
        return crate::historical::profile_event(r, options, &time, execute);
    }
    let mut o: Options = serde_json::from_value(options.cloned().unwrap_or_else(|| json!({})))
        .map_err(|e| invalid(&e.to_string()))?;
    let mut unique = std::collections::HashSet::new();
    if o.covariates.len() > 32
        || o.covariates
            .iter()
            .any(|n| n.is_empty() || !unique.insert(n))
        || !["ode_fixed", "ode_historical_batch"].contains(&o.prediction_mode.as_str())
        || !["corrected", "historical"].contains(&o.coefficient_layout.as_str())
        || !["physical", "historical"].contains(&o.effective_domain.as_str())
        || !["raw", "cumulative_max"].contains(&o.projection.as_str())
        || o.t_event.is_some_and(|t| !t.is_finite())
    {
        return Err(invalid(
            "invalid ODE mode, declared covariates, event, layout, domain or projection",
        ));
    }
    let historical = o.prediction_mode == "ode_historical_batch";
    if historical && (o.initial_time.is_some() || o.initial_value.is_some()) {
        return Err(invalid(
            "historical batch mode fixes the historical initial value at each requested start",
        ));
    }
    let origin = if historical {
        time.first().copied().unwrap_or(0.)
    } else {
        o.initial_time.unwrap_or(0.)
    };
    let initial = o
        .initial_value
        .unwrap_or(if key == "bass" { 0. } else { 1e-6 });
    if !origin.is_finite()
        || origin < 0.
        || !initial.is_finite()
        || initial < 0.
        || time.first().is_some_and(|t| *t < origin)
    {
        return Err(invalid("invalid initial time/value"));
    }
    if !historical {
        o.initial_time = Some(origin);
        o.initial_value = Some(initial);
    }
    let forcing = History::merge(
        o.covariate_history.take(),
        &o.covariates,
        &time,
        input.get("covariates"),
        origin,
    )
    .map_err(|e| invalid(&e))?;
    o.covariate_history = forcing.clone();
    let mut spec = vec![];
    for name in base {
        spec.push(Parameter {
            name: name.into(),
            positive: true,
            capacity: name == "m" || name == "a",
            unused: key == "gompertz" && name == "b",
        });
    }
    if o.t_event.is_some() {
        for name in base {
            spec.push(Parameter {
                name: format!("{name}_post"),
                positive: true,
                capacity: name == "m" || name == "a",
                unused: key == "gompertz" && name == "b",
            });
        }
    }
    for cov in &o.covariates {
        for name in base {
            spec.push(Parameter {
                name: format!("beta_{name}_{cov}"),
                positive: false,
                capacity: name == "m" || name == "a",
                unused: key == "gompertz" && name == "b",
            });
        }
    }
    let model = Model {
        key: key.into(),
        options: o,
        spec,
        origin,
        initial,
        forcing,
    };
    let observed = if op == KernelOperation::FitModel
        || op == KernelOperation::DiagnoseModel
        || input.contains_key("observed")
        || input.contains_key("y")
    {
        let y = numeric_array_from_aliases(input, &["observed", "y"], op)?;
        if y.len() != time.len() || y.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(invalid(
                "observations must be finite nonnegative and aligned",
            ));
        }
        Some(y)
    } else {
        None
    };
    let active: Vec<_> = model
        .spec
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.unused)
        .map(|(i, _)| i)
        .collect();
    let weights = if op == KernelOperation::FitModel {
        crate::fitting::weights(r, time.len(), active.len().max(4))?
    } else {
        None
    };
    let max_weight = weights
        .as_ref()
        .map_or(1., |w| w.iter().copied().fold(0., f64::max));
    let factors: Vec<_> = weights.as_ref().map_or_else(
        || vec![1.; time.len()],
        |w| w.iter().map(|v| v.sqrt() / max_weight.sqrt()).collect(),
    );
    let mut quality = None;
    let parameters = if op == KernelOperation::FitModel {
        let y = observed.as_ref().unwrap();
        if time.len() < 4 {
            return Err(invalid("ODE fitting needs at least four observations"));
        }
        let (initial, limit) = crate::fitting::scalar_options(r)?;
        let ys = y
            .iter()
            .zip(&factors)
            .filter(|(_, w)| **w > 0.)
            .map(|(v, _)| *v)
            .fold(0., f64::max)
            .max(1e-12);
        let ts = (time.last().unwrap() - origin).max(1.);
        let automatic_seed = |rate: f64| -> Vec<f64> {
            model
                .spec
                .iter()
                .map(|parameter| {
                    if !parameter.positive {
                        0.
                    } else if parameter.capacity {
                        1.25 * ys
                    } else if parameter.unused {
                        1.
                    } else if parameter.name == "p" || parameter.name == "p_post" {
                        0.5 / ts
                    } else {
                        rate / ts
                    }
                })
                .collect()
        };
        let seed = initial
            .as_ref()
            .map(|value| model.parse(value))
            .transpose()
            .map_err(|e| invalid(&e))?
            .unwrap_or_else(|| automatic_seed(4.));
        let scales: Vec<_> = model
            .spec
            .iter()
            .map(|s| if s.capacity { ys } else { 1. / ts })
            .collect();
        let transform = |p: &[f64]| {
            active
                .iter()
                .map(|i| {
                    if model.spec[*i].positive {
                        (p[*i] / scales[*i]).max(1e-12).ln()
                    } else {
                        p[*i] / scales[*i]
                    }
                })
                .collect::<Vec<_>>()
        };
        let untransform = |theta: &[f64]| {
            let mut p = seed.clone();
            for (i, t) in active.iter().zip(theta) {
                p[*i] = if model.spec[*i].positive {
                    t.exp() * scales[*i]
                } else {
                    t * scales[*i]
                };
            }
            p
        };
        let starts = if initial.is_some() {
            let theta = transform(&seed);
            vec![
                theta.clone(),
                theta.iter().map(|t| t + 0.1).collect(),
                theta.iter().map(|t| t - 0.1).collect(),
            ]
        } else {
            [1., 4., 8.]
                .into_iter()
                .map(|rate| transform(&automatic_seed(rate)))
                .collect()
        };
        let target: Vec<_> = y.iter().zip(&factors).map(|(v, w)| (v * w) / ys).collect();
        let fit = least_squares(
            &target,
            &starts,
            |theta| {
                let prediction = model.predict(&time, &untransform(theta))?;
                prediction["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(&factors)
                    .map(|(v, w)| {
                        v.as_f64()
                            .map(|v| (v * w) / ys)
                            .ok_or_else(|| "nonfinite ODE prediction".into())
                    })
                    .collect()
            },
            limit,
        )
        .map_err(|e| invalid(&e))?;
        let p = untransform(&fit.parameters);
        quality = Some((fit.report, fit.identifiability));
        p
    } else {
        model
            .parse(
                p.get("state")
                    .and_then(|s| s.get("parameters"))
                    .or_else(|| p.get("parameters"))
                    .ok_or_else(|| invalid("parameters required"))?,
            )
            .map_err(|e| invalid(&e))?
    };
    let eval = if time.is_empty() {
        vec![origin]
    } else {
        time.clone()
    };
    let predictions = model.predict(&eval, &parameters).map_err(|e| invalid(&e))?;
    let state = json!({"model_key":key,"parameters":model.value(&parameters),"constructor_kwargs":model.options,"predict_kwargs":{}});
    let absent: Vec<_> = model
        .spec
        .iter()
        .filter(|s| s.unused)
        .map(|s| s.name.clone())
        .collect();
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        let mut response = predictions;
        response["state"] = state;
        response
    } else {
        let mut out = json!({"model_key":key,"model_name":if key=="bass"{"MechanisticBassModel"}else{"MechanisticGompertzModel"},"parameters":model.value(&parameters),"state":state,"predictions":predictions,"details":{"initial_time":origin,"initial_value":initial,"absent_from_rhs":absent}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if let Some(y) = observed {
            let predicted: Vec<_> = predictions["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let mut d = crate::kernel::fit_diagnostics_count(
                &time,
                &y,
                &predicted,
                out["model_name"].as_str().unwrap(),
                active.len() + 1,
            );
            d["metrics"]["AIC"] = Value::Null;
            d["metrics"]["BIC"] = Value::Null;
            d["information_criteria"] = json!({"available":false,"reason":"conditional forcing/initial-state and projection likelihood require qualification"});
            if weights.is_some() {
                let weighted_rss: f64 = y
                    .iter()
                    .zip(&predicted)
                    .zip(&factors)
                    .map(|((y, prediction), factor)| (y * factor - prediction * factor).powi(2))
                    .sum();
                d["weighting"] = json!({
                    "maximum_weight": max_weight,
                    "positive_observations": factors.iter().filter(|factor| **factor > 0.).count(),
                    "weighted_rss_normalized_by_max": weighted_rss,
                    "rank_semantics": "weighted local Jacobian; zero-weight residuals excluded",
                    "descriptive_metrics": "unweighted"
                });
            }
            d["structural_identifiability"] =
                json!({"absent_from_rhs":absent,"absent_parameters_held_fixed":true});
            d["warnings"].as_array_mut().unwrap().push(json!("Forcing history and initial condition are treated as fixed; their uncertainty is not propagated. Gompertz b/beta_b terms do not enter the ODE. Projection and historical coefficient layout are explicit conventions."));
            if let Some((optimizer, mut rank)) = quality {
                if optimizer["converged"] != true {
                    d["warnings"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!("optimizer did not converge"));
                }
                if rank["status"] != "local_full_rank" {
                    d["warnings"].as_array_mut().unwrap().push(json!(
                        "active forcing parameters are locally rank deficient"
                    ));
                }
                rank["scope"] = json!(
                    "active ODE parameters only; not global uniqueness or calibrated uncertainty"
                );
                d["optimizer"] = optimizer;
                d["identifiability"] = rank;
            }
            out["diagnostics"] = d;
        }
        out
    };
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: r.model_key.clone(),
        result: Some(result),
        error: None,
        metadata: json!({"runtime":"rust_native","family":key,"execution":"mechanistic_ode"}),
    })
}
