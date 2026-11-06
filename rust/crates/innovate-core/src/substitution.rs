//! Multi-generation substitution. Historical ODE and successive-generation semantics are explicit.
use crate::numerics::{integrate, least_squares};
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
    for (key, value) in kw {
        if key == "covariates" && value.as_array().is_some_and(Vec::is_empty) {
            continue;
        }
        if ![
            "n_generations",
            "variant",
            "launch_times",
            "initial_time",
            "initial_values",
        ]
        .contains(&key.as_str())
        {
            return Err(Error::unsupported_native_operation(
                op,
                format!("Norton-Bass option '{key}' is not implemented"),
            ));
        }
    }
    if inputs.contains_key("covariates") {
        return Err(Error::unsupported_native_operation(
            op,
            "Norton-Bass covariates remain a migration obligation",
        ));
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
    Ok(Model {
        n,
        variant,
        launches,
        origin,
        initial,
    })
}

fn parameters(value: &Value, model: &Model, op: KernelOperation) -> Result<Vec<f64>, Error> {
    let p = value
        .as_object()
        .ok_or_else(|| invalid(op, "parameters must be an object"))?;
    if p.len() != 3 * model.n {
        return Err(invalid(
            op,
            "parameters must contain exactly p_i, q_i, m_i for each generation",
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
    Ok(result)
}
fn parameter_value(p: &[f64]) -> Value {
    let mut out = Map::new();
    for (i, p) in p.as_chunks::<3>().0.iter().enumerate() {
        for (key, value) in ['p', 'q', 'm'].into_iter().zip(p) {
            out.insert(format!("{key}{}", i + 1), json!(value));
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
    fn kwargs(&self) -> Value {
        json!({"n_generations":self.n,"variant":self.variant,"launch_times":self.launches,"initial_time":self.origin,"initial_values":self.initial})
    }
    fn derivative(&self, y: &[f64], p: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                (p[3 * i] + p[3 * i + 1] * y[i] / p[3 * i + 2])
                    * (p[3 * i + 2] - y[i..].iter().sum::<f64>())
            })
            .collect()
    }
    fn predict(&self, time: &[f64], p: &[f64]) -> Result<Trajectory, String> {
        if self.variant == "legacy_cannibalization" {
            let rows = if self.n == 1 && self.initial[0] == 0.0 {
                if time.iter().any(|t| *t < self.origin) {
                    return Err("time precedes initial_time".into());
                }
                time.iter()
                    .map(|t| vec![p[2] * fraction(t - self.origin, p[0], p[1]).0])
                    .collect()
            } else {
                integrate(time, self.origin, &self.initial, |_, y| {
                    self.derivative(y, p)
                })?
            };
            let rates: Vec<Vec<f64>> = rows.iter().map(|y| self.derivative(y, p)).collect();
            if rows.iter().chain(&rates).flatten().any(|x| !x.is_finite()) {
                return Err("prediction exceeded the finite numerical domain".into());
            }
            return Ok((rows, rates));
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
    fn prediction_value(&self, time: &[f64], p: &[f64]) -> Result<Value, String> {
        let (rows, rates) = self.predict(time, p)?;
        let warnings = if rows.iter().flatten().any(|x| *x < 0.0) {
            vec![
                "Historical cannibalization equations produced negative levels; inspect the model/market assumptions",
            ]
        } else {
            vec![]
        };
        Ok(
            json!({"columns":(1..=self.n).map(|i|format!("series_{i}")).collect::<Vec<_>>(),"rows":rows,"rates":rates,
            "metadata":{"shape":[time.len(),self.n],"variant":self.variant,"quantity":"generation_level",
            "rate_semantics":"signed derivative of generation level; not gross incidence", "launch_derivative":"right_hand", "warnings":warnings}}),
        )
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
        .map(|i| observed.iter().map(|r| r[i]).sum::<f64>() / observed.len() as f64)
        .collect();
    let total: f64 = observed
        .iter()
        .flat_map(|row| row.iter().zip(&means).map(|(y, m)| (y - m).powi(2)))
        .sum();
    let rss = d["metrics"]["RSS"].as_f64().unwrap();
    let r2 = if total > 0.0 {
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
    let model = options(request, inputs)?;
    let time = numeric_array_from_aliases(inputs, &["time", "t"], op)?;
    if time.iter().any(|t| !t.is_finite() || *t < 0.0) || time.windows(2).any(|t| t[1] <= t[0]) {
        return Err(invalid(
            op,
            "time must be finite, nonnegative and strictly increasing",
        ));
    }
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
        for p0 in [0.1, 0.5, 1.0] {
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
    let predictions = model
        .prediction_value(&time, &p)
        .map_err(|e| invalid(op, e))?;
    let state = json!({"model_key":"norton_bass","model_name":"NortonBassModel","parameters":parameter_value(&p),"constructor_kwargs":model.kwargs(),"predict_kwargs":{}});
    let result = match op {
        KernelOperation::PredictModel | KernelOperation::SimulateModel => predictions,
        _ => {
            let mut result = json!({"model_key":"norton_bass","model_name":"NortonBassModel","family":"substitution","parameters":parameter_value(&p),"state":state,"predictions":predictions});
            if op != KernelOperation::SummarizeModel
                || inputs.contains_key("observed")
                || inputs.contains_key("y")
            {
                let observed = observations(inputs, model.n, time.len(), op)?;
                let (predicted, _) = model.predict(&time, &p).map_err(|e| invalid(op, e))?;
                let mut d =
                    diagnostics(&time, &observed, &predicted, "NortonBassModel", 3 * model.n);
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
