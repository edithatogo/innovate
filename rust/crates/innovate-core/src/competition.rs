//! Native competitive diffusion equations preserved from the pre-migration models.
use crate::events::ParameterEvents;
use crate::mechanistic::History;
use crate::numerics::least_squares;
use crate::validation::{numeric_array, numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use serde_json::{Map, Value, json};

struct Model {
    key: String,
    n: usize,
    names: Vec<String>,
    base_count: usize,
    covariates: Vec<String>,
    history: Option<History>,
    historical_layout: bool,
    events: ParameterEvents,
    origin: f64,
    initial: Vec<f64>,
    conditioned_initial: bool,
}
impl Model {
    fn name(&self) -> &str {
        if self.key == "lock_in" {
            "LockInModel"
        } else if self.key == "complementary_goods" {
            "ComplementaryGoodsModel"
        } else if self.key == "lotka_volterra" {
            "LotkaVolterraModel"
        } else {
            "MultiProductDiffusionModel"
        }
    }
    fn kwargs(&self) -> Value {
        let mut out = json!({"initial_time":self.origin,"initial_values":self.initial});
        if self.key == "multi_product" {
            out["n_products"] = json!(self.n);
        }
        if let Some(history) = &self.history {
            out["covariates"] = json!(self.covariates);
            out["covariate_history"] = json!(history);
            if self.key == "multi_product" {
                out["coefficient_layout"] = json!(if self.historical_layout {
                    "historical"
                } else {
                    "corrected"
                });
            }
        }
        if !self.events.is_empty() {
            out["parameter_events"] = self.events.value();
        }
        out
    }
    fn parameters(&self, v: &Value, op: KernelOperation) -> Result<Vec<f64>, KernelBindingError> {
        let p = v
            .as_object()
            .ok_or_else(|| invalid(op, "parameters must be an object"))?;
        if p.len() != self.names.len() {
            return Err(invalid(
                op,
                "parameters must match the model's complete named parameter set",
            ));
        }
        self.names.iter().enumerate().map(|(index, name)| {
            let x = required_f64(p,name,op)?;
            if !x.is_finite() || (index < self.base_count && (x < 0. || (name.starts_with('m') && x == 0.))) {
                return Err(invalid(op,"rates/interactions must be nonnegative; market potentials must be positive"));
            }
            Ok(x)
        }).collect()
    }
    fn value(&self, p: &[f64]) -> Value {
        Value::Object(
            self.names
                .iter()
                .zip(p)
                .map(|(name, p)| (name.clone(), json!(p)))
                .collect(),
        )
    }
    fn regime(&self, t: f64) -> usize {
        self.events.regime(t)
    }
    fn lock_in_capacity(&self, t: f64, parameters: &[f64], regime: usize) -> f64 {
        let parameters = self.events.parameters(parameters, regime);
        parameters[6]
            + self.history.as_ref().map_or(0., |history| {
                self.covariates
                    .iter()
                    .enumerate()
                    .map(|(index, name)| {
                        parameters[self.base_count * (index + 1) + 6] * history.at(name, t)
                    })
                    .sum::<f64>()
            })
    }

    fn derivative(&self, t: f64, y: &[f64], parameters: &[f64]) -> Vec<f64> {
        self.derivative_in_regime(t, y, parameters, self.regime(t))
    }
    fn derivative_in_regime(
        &self,
        t: f64,
        y: &[f64],
        parameters: &[f64],
        regime: usize,
    ) -> Vec<f64> {
        let parameters = self.events.parameters(parameters, regime);
        if let Some(history) = &self.history {
            let mut effective = parameters[..self.base_count].to_vec();
            for (name, coefficients) in self
                .covariates
                .iter()
                .zip(parameters[self.base_count..].chunks_exact(self.base_count))
            {
                let value = history.at(name, t);
                for (index, parameter) in effective.iter_mut().enumerate() {
                    let coefficient_index = if self.historical_layout && index < 3 * self.n {
                        3 * (index % self.n) + index / self.n
                    } else {
                        index
                    };
                    *parameter += coefficients[coefficient_index] * value;
                }
            }
            if effective.iter().any(|value| !value.is_finite())
                || (self.key == "lock_in" && effective[6] <= 0.)
            {
                return vec![f64::NAN; self.n];
            }
            self.base_derivative(y, &effective)
        } else {
            self.base_derivative(y, &parameters)
        }
    }
    fn base_derivative(&self, y: &[f64], p: &[f64]) -> Vec<f64> {
        if self.key == "lock_in" {
            let m = p[6];
            let a = y[0].clamp(0., m);
            let b = y[1].clamp(0., m);
            return vec![
                p[0] * a * (1. - (a + b) / m) + p[2] * a * a / m - p[4] * a * b / m,
                p[1] * b * (1. - (a + b) / m) + p[3] * b * b / m - p[5] * b * a / m,
            ];
        }

        if self.key == "complementary_goods" {
            return vec![
                p[0] * y[0] * (1. - y[0]) + p[2] * y[0] * y[1],
                p[1] * y[1] * (1. - y[1]) + p[3] * y[0] * y[1],
            ];
        }
        if self.key == "lotka_volterra" {
            return vec![
                p[0] * y[0] * (1. - y[0]) - p[1] * y[0] * y[1],
                p[2] * y[1] * (1. - y[1]) - p[3] * y[0] * y[1],
            ];
        }
        (0..self.n)
            .map(|i| {
                if p[2 * self.n + i] <= 0. {
                    return 0.;
                }
                let interactions: f64 = (0..self.n)
                    .filter(|&j| j != i)
                    .enumerate()
                    .map(|(offset, j)| p[3 * self.n + i * (self.n - 1) + offset] * y[j])
                    .sum();
                (p[i] + p[self.n + i] * y[i] / p[2 * self.n + i])
                    * (p[2 * self.n + i] - y[i] - interactions)
            })
            .collect()
    }
    fn predict(&self, time: &[f64], p: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        let end = *time.last().ok_or("time must not be empty")?;
        let mut grid = time.to_vec();
        if let Some(history) = &self.history {
            grid.extend(
                history
                    .time
                    .iter()
                    .copied()
                    .filter(|t| *t > self.origin && *t < end),
            );
        }
        grid.extend(
            self.events
                .times()
                .filter(|t| *t > self.origin && *t <= end),
        );
        grid.sort_by(f64::total_cmp);
        grid.dedup_by(|a, b| *a == *b);
        if self.key == "lock_in" {
            for time in std::iter::once(self.origin).chain(grid.iter().copied()) {
                let active = self.regime(time);
                let left = self
                    .events
                    .boundary_left_regime(time)
                    .filter(|_| time > self.origin);
                for regime in std::iter::once(active).chain(left) {
                    let capacity = self.lock_in_capacity(time, p, regime);
                    if !capacity.is_finite() || capacity <= 0. {
                        return Err(
                            "lock-in effective capacity must stay finite and positive".into()
                        );
                    }
                }
            }
        }
        let states = self
            .events
            .integrate(&grid, self.origin, &self.initial, |t, y, regime| {
                self.derivative_in_regime(t, y, p, regime)
            })?;
        let mut rows: Vec<_> = time
            .iter()
            .map(|t| states[grid.partition_point(|v| v < t)].clone())
            .collect();
        if self.key == "lock_in" {
            for (&t, row) in time.iter().zip(&mut rows) {
                let market = self.lock_in_capacity(t, p, self.regime(t));
                for value in row {
                    *value = value.clamp(0., market);
                }
            }
        }
        Ok(rows)
    }
    fn prediction(&self, time: &[f64], p: &[f64], rows: &[Vec<f64>]) -> Result<Value, String> {
        let rates: Vec<_> = time
            .iter()
            .zip(rows)
            .map(|(&t, y)| self.derivative(t, y, p))
            .collect();
        if rows.iter().chain(&rates).flatten().any(|x| !x.is_finite()) {
            return Err("prediction exceeded the finite numerical domain".into());
        }
        let warnings = if rows.iter().flatten().any(|x| *x < 0.) {
            vec![
                "Historical competitive equations produced negative levels; inspect the interaction assumptions",
            ]
        } else {
            vec![]
        };
        Ok(
            json!({"columns":(1..=self.n).map(|i|format!("series_{i}")).collect::<Vec<_>>(),"rows":rows,"rates":rates,
            "metadata":{"shape":[time.len(),self.n],"quantity":if self.key=="lotka_volterra" {"technology_share"}else{"product_level"},
            "rate_semantics":if self.key=="lock_in" {"historical RHS at clipped levels; not derivative of output clipping or gross incidence"} else {"signed derivative of each level; not gross incidence"},"warnings":warnings,
            "normalization":"separate product potentials; rows are not constrained to sum to one",
            "parameter_event_semantics":"persistent fixed overrides; continuous integrated state; right-hand rates; shared covariate history",
            "projection":if self.key=="lock_in" {"historical_clip_to_current_capacity"} else {"none"},
            "coefficient_layout":if self.historical_layout {"historical"} else {"corrected"}}}),
        )
    }
}
fn invalid(op: KernelOperation, message: impl Into<String>) -> KernelBindingError {
    KernelBindingError::invalid_request(op, message)
}

fn model(
    request: &KernelRequest,
    inputs: &Map<String, Value>,
    time: &[f64],
) -> Result<Model, KernelBindingError> {
    let op = request.operation;
    let p = request.payload.as_object().unwrap();
    let key = request.model_key.clone().unwrap();
    let empty = json!({});
    let kw = p
        .get("constructor_kwargs")
        .or_else(|| p.get("model_kwargs"))
        .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")))
        .unwrap_or(&empty)
        .as_object()
        .ok_or_else(|| invalid(op, "constructor_kwargs must be an object"))?;
    if kw.keys().any(|key| {
        ![
            "n_products",
            "initial_time",
            "initial_values",
            "covariates",
            "covariate_history",
            "coefficient_layout",
            "parameter_events",
        ]
        .contains(&key.as_str())
    }) {
        return Err(KernelBindingError::unsupported_native_operation(
            op,
            "unknown competitive option",
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
    let unique: std::collections::HashSet<_> = covariates.iter().collect();
    if covariates.len() > 32
        || unique.len() != covariates.len()
        || covariates.iter().any(|name| name.trim().is_empty())
    {
        return Err(invalid(
            op,
            "covariates must contain at most 32 distinct nonempty names",
        ));
    }
    let historical_layout = match kw.get("coefficient_layout").and_then(Value::as_str) {
        Some("historical") if key == "multi_product" => true,
        Some("corrected") if key == "multi_product" => false,
        None if !kw.contains_key("coefficient_layout") => false,
        _ => {
            return Err(invalid(
                op,
                "coefficient_layout requires corrected or historical on multi_product",
            ));
        }
    };
    if ["lotka_volterra", "complementary_goods", "lock_in"].contains(&key.as_str())
        && kw.contains_key("n_products")
    {
        return Err(invalid(op, "Lotka-Volterra has exactly two technologies"));
    }
    if p.get("state")
        .and_then(|s| s.get("model_key"))
        .and_then(Value::as_str)
        .is_some_and(|m| m != key)
    {
        return Err(invalid(op, "request and state model_key differ"));
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
            return Err(KernelBindingError::unsupported_native_operation(
                op,
                "custom prediction/fitter options are unavailable",
            ));
        }
    }
    let observed = inputs.get("observed").or_else(|| inputs.get("y"));
    let n = if ["lotka_volterra", "complementary_goods", "lock_in"].contains(&key.as_str()) {
        2
    } else if let Some(n) = kw.get("n_products") {
        n.as_u64()
            .ok_or_else(|| invalid(op, "n_products must be a positive integer"))? as usize
    } else if let Some(p) = p
        .get("state")
        .and_then(|s| s.get("parameters"))
        .and_then(Value::as_object)
    {
        p.keys()
            .filter(|k| {
                k.strip_prefix('p')
                    .is_some_and(|i| i.parse::<usize>().is_ok())
            })
            .count()
    } else {
        observed
            .and_then(Value::as_array)
            .and_then(|r| r.first())
            .and_then(Value::as_array)
            .map_or(1, Vec::len)
    };
    if n == 0 || n > 32 {
        return Err(invalid(op, "n_products must be in 1..=32"));
    }
    let inferred_origin = if op == KernelOperation::FitModel {
        time[0]
    } else {
        0.
    };
    let origin = kw
        .get("initial_time")
        .map(|v| {
            v.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| invalid(op, "initial_time must be finite"))
        })
        .transpose()?
        .unwrap_or(inferred_origin);
    if origin < 0. || origin > time[0] {
        return Err(invalid(
            op,
            "initial_time must be nonnegative and no later than the first time",
        ));
    }
    let initial = if let Some(y) = kw.get("initial_values") {
        numeric_array(y, "initial_values", op)?
    } else if op == KernelOperation::FitModel && origin == time[0] {
        let first = observed
            .and_then(Value::as_array)
            .and_then(|r| r.first())
            .ok_or_else(|| invalid(op, "observed is required"))?;
        if n == 1 && first.is_number() {
            vec![first.as_f64().unwrap()]
        } else {
            numeric_array(first, "initial observed row", op)?
        }
    } else if key == "multi_product" {
        vec![0.; n]
    } else {
        return Err(invalid(
            op,
            "Lotka-Volterra prediction requires explicit initial_values",
        ));
    };
    let conditioned_initial = (op == KernelOperation::FitModel
        && origin == time[0]
        && !kw.contains_key("initial_values"))
        || p.get("state")
            .and_then(|s| s.get("fit_context"))
            .and_then(|s| s.get("conditioned_time"))
            .and_then(Value::as_f64)
            == Some(origin);
    if initial.len() != n
        || initial
            .iter()
            .any(|v| !v.is_finite() || *v < 0. || (key == "lotka_volterra" && *v > 1.))
    {
        return Err(invalid(
            op,
            "initial_values must match the model dimensions and nonnegative domain (shares <=1)",
        ));
    }
    let mut names: Vec<String> = if key == "lock_in" {
        [
            "alpha1", "alpha2", "beta1", "beta2", "gamma1", "gamma2", "m",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    } else if key == "complementary_goods" {
        ["k1", "k2", "c1", "c2"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else if key == "lotka_volterra" {
        ["alpha1", "beta1", "alpha2", "beta2"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        let mut names: Vec<_> = ['p', 'q', 'm']
            .into_iter()
            .flat_map(|prefix| (1..=n).map(move |i| format!("{prefix}{i}")))
            .collect();
        for i in 1..=n {
            for j in 1..=n {
                if i != j {
                    names.push(format!("alpha_{i}_{j}"));
                }
            }
        }
        names
    };
    let base_count = names.len();
    for name in &covariates {
        for index in 0..base_count {
            names.push(format!("beta_{}_{}", names[index], name));
        }
    }
    let history = kw
        .get("covariate_history")
        .map(|value| {
            serde_json::from_value(value.clone())
                .map_err(|_| invalid(op, "invalid covariate_history"))
        })
        .transpose()?;
    let history = History::merge(history, &covariates, time, inputs.get("covariates"), origin)
        .map_err(|e| invalid(op, e))?;
    let events = ParameterEvents::parse(kw.get("parameter_events"), &names, base_count)
        .map_err(|error| invalid(op, error))?;
    Ok(Model {
        key,
        n,
        names,
        base_count,
        covariates,
        history,
        historical_layout,
        events,
        origin,
        initial,
        conditioned_initial,
    })
}

pub(crate) fn execute(request: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = request.operation;
    let payload = request
        .payload
        .as_object()
        .ok_or_else(|| invalid(op, "payload must be an object"))?;
    let inputs = object_section(payload, "inputs", op)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], op)?;
    if time.iter().any(|t| !t.is_finite() || *t < 0.) || time.windows(2).any(|t| t[1] <= t[0]) {
        return Err(invalid(
            op,
            "time must be finite, nonnegative and strictly increasing",
        ));
    }
    let model = model(request, inputs, &time)?;
    let mut quality = None;
    let params = if op == KernelOperation::FitModel {
        if time.len() < 4 {
            return Err(invalid(op, "fitting requires at least four time points"));
        }
        let observed = crate::substitution::observations(inputs, model.n, time.len(), op)?;
        if observed.iter().flatten().any(|x| *x < 0.) {
            return Err(invalid(
                op,
                "competitive fitting requires nonnegative observations",
            ));
        }
        let yscale = observed.iter().flatten().copied().fold(0., f64::max);
        if yscale == 0. {
            return Err(invalid(
                op,
                "all-zero observations cannot identify competitive parameters",
            ));
        }
        let tscale = (time[time.len() - 1] - model.origin).max(1.);
        let scales: Vec<_> = (0..model.names.len())
            .map(|index| {
                let base_name = &model.names[index % model.base_count];
                if base_name.starts_with('m') {
                    yscale
                } else if base_name.starts_with("alpha_") {
                    1.
                } else {
                    1. / tscale
                }
            })
            .collect();
        let offsets: Vec<f64> = model
            .names
            .iter()
            .map(|name| {
                if model.key == "lock_in"
                    && model.history.is_none()
                    && model.events.is_empty()
                    && name == "m"
                {
                    yscale
                } else {
                    0.
                }
            })
            .collect();
        let transform = |p: Vec<f64>| {
            p.iter()
                .zip(&scales)
                .zip(&offsets)
                .enumerate()
                .map(|(index, ((p, s), offset))| {
                    if index >= model.base_count {
                        *p / s
                    } else {
                        ((p - offset) / s).max(1e-12).ln()
                    }
                })
                .collect::<Vec<_>>()
        };
        let untransform = |theta: &[f64]| {
            theta
                .iter()
                .zip(&scales)
                .zip(&offsets)
                .enumerate()
                .map(|(index, ((t, s), offset))| {
                    if index >= model.base_count {
                        t * s
                    } else {
                        t.exp() * s + offset
                    }
                })
                .collect::<Vec<_>>()
        };
        let (initial, limit) = crate::fitting::options(request)?;
        let mut starts = Vec::new();
        if let Some(initial) = initial {
            let parameters = model.parameters(&initial, op)?;
            if model.key == "lock_in"
                && model.history.is_none()
                && model.events.is_empty()
                && parameters[6] < yscale
            {
                return Err(invalid(
                    op,
                    "initial market potential must be at least the maximum observed value across both series",
                ));
            }
            starts.push(transform(parameters));
        }
        if model.history.is_some() && starts.is_empty() {
            return Err(invalid(
                op,
                "covariate fitting requires complete initial_parameters",
            ));
        }
        for rate in if model.history.is_none() {
            &[0.5, 2., 5.][..]
        } else {
            &[]
        } {
            let p = model
                .names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if model.key == "lock_in"
                        && model.history.is_none()
                        && model.events.is_empty()
                        && name == "m"
                    {
                        yscale * 1.5
                    } else if name.starts_with('m') {
                        observed
                            .iter()
                            .map(|r| {
                                r[if model.key == "lock_in" {
                                    0
                                } else {
                                    i - 2 * model.n
                                }]
                            })
                            .fold(0., f64::max)
                            .max(yscale * 0.1)
                            * 1.2
                    } else if name.starts_with("alpha_") {
                        0.1
                    } else if name.starts_with('p')
                        || name.starts_with("beta")
                        || name.starts_with('c')
                    {
                        0.1 / tscale
                    } else {
                        rate / tscale
                    }
                })
                .collect();
            starts.push(transform(p));
        }
        let target: Vec<_> = observed.iter().flatten().map(|x| x / yscale).collect();
        let fit = least_squares(
            &target,
            &starts,
            |theta| {
                model
                    .predict(&time, &untransform(theta))
                    .map(|r| r.into_iter().flatten().map(|x| x / yscale).collect())
            },
            limit,
        )
        .map_err(|e| invalid(op, e))?;
        let p = untransform(&fit.parameters);
        quality = Some((fit.report, fit.identifiability));
        p
    } else {
        model.parameters(
            payload
                .get("parameters")
                .or_else(|| payload.get("state").and_then(|s| s.get("parameters")))
                .ok_or_else(|| invalid(op, "parameters are required"))?,
            op,
        )?
    };
    let rows = model.predict(&time, &params).map_err(|e| invalid(op, e))?;
    let predictions = model
        .prediction(&time, &params, &rows)
        .map_err(|e| invalid(op, e))?;
    let parameters = model.value(&params);
    let state = json!({"model_key":model.key,"model_name":model.name(),"parameters":parameters,"constructor_kwargs":model.kwargs(),"predict_kwargs":{},
        "fit_context":{"conditioned_time":if model.conditioned_initial {Some(model.origin)} else {None}}});
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        let mut predictions = predictions;
        if model.history.is_some() || !model.events.is_empty() {
            predictions["state"] = state;
        }
        predictions
    } else {
        let mut r = json!({"model_key":model.key,"model_name":model.name(),"family":"competition","parameters":parameters,"state":state,"predictions":predictions});
        if op != KernelOperation::SummarizeModel
            || inputs.contains_key("observed")
            || inputs.contains_key("y")
        {
            let observed = crate::substitution::observations(inputs, model.n, time.len(), op)?;
            let offset = usize::from(model.conditioned_initial && time[0] == model.origin);
            if time.len() <= offset {
                return Err(invalid(
                    op,
                    "diagnostics require observations after the conditioning time",
                ));
            }
            let mut d = crate::substitution::diagnostics(
                &time[offset..],
                &observed[offset..],
                &rows[offset..],
                model.name(),
                model.names.len(),
            );
            d["initial_condition"] = json!({"source":if model.conditioned_initial {"first_observed_row"} else {"explicit_or_zero"},"excluded_conditioning_rows":offset,"uncertainty_propagated":false});
            if model.key == "lock_in" && model.events.is_empty() {
                d["structural_identifiability"] = json!({"status":"interior_parameter_confounding","scope":"unclipped_interior_trajectories","rank_upper_bound":model.names.len()-1,"parameters":model.names.len(),"reason":"a common capacity and feedback scaling preserves the unclipped interior RHS; clipping can add boundary information"});
                d["warnings"].as_array_mut().unwrap().push(json!("Interior trajectories cannot identify free m separately from beta/gamma combinations; clipping-boundary information requires separate qualification"));
            }
            if model.key == "lock_in" && !model.events.is_empty() {
                d["structural_identifiability"] = json!({"status":"requires_event_specific_analysis","parameters":model.names.len(),
                    "rank_upper_bound":null,"reason":"fixed event overrides can break shared interior scaling; local numerical rank alone is insufficient"});
                d["warnings"].as_array_mut().unwrap().push(json!("Fixed event parameter constraints require event-specific structural identifiability analysis"));
            }
            if !model.events.is_empty() {
                d["parameter_events"] = json!({"count":model.events.len(),"parameters_estimated":false,"state_reset":false});
            }
            if let Some((optimizer, identifiability)) = quality {
                if optimizer["converged"] != true {
                    d["warnings"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!("optimizer did not converge"));
                }
                if identifiability["status"] != "local_full_rank" {
                    d["warnings"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!("parameters are locally rank deficient"));
                }
                d["optimizer"] = optimizer;
                d["identifiability"] = identifiability;
            }
            r["diagnostics"] = d;
        }
        r
    };
    Ok(KernelResponse {
        schema_version: KERNEL_SCHEMA_VERSION.into(),
        operation: op,
        model_key: Some(model.key.clone()),
        error: None,
        result: Some(result),
        metadata: json!({"model_key":model.key,"model_name":model.name(),"family":"competition","runtime":"rust_native"}),
    })
}
