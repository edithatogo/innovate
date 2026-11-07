//! Heterogeneous scalar growth ODEs with signed linear cross-product interactions.
use crate::numerics::{integrate, least_squares};
use crate::validation::{numeric_array_from_aliases, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Options {
    components: Vec<String>,
    #[serde(default = "capacity_bounds")]
    capacity_bounds: String,
    #[serde(default = "origin_mode")]
    origin_mode: String,
    #[serde(default)]
    initial_time: Option<f64>,
    #[serde(default)]
    initial_values: Option<Vec<f64>>,
    #[serde(default)]
    alpha: Option<Vec<Vec<f64>>>,
}
fn capacity_bounds() -> String {
    "positive".into()
}
fn origin_mode() -> String {
    "fixed".into()
}
fn names(model: &str) -> Option<&'static [&'static str]> {
    match model {
        "bass" => Some(&["p", "q", "m"]),
        "logistic" => Some(&["L", "k", "x0"]),
        "gompertz" => Some(&["a", "b", "c"]),
        "fisher_pry" => Some(&["alpha", "t0"]),
        _ => None,
    }
}
struct Parameter {
    name: String,
    positive: bool,
    unused: bool,
    capacity: Option<usize>,
}
struct Model {
    options: Options,
    parameters: Vec<Parameter>,
    starts: Vec<usize>,
    base_count: usize,
    origin: f64,
    initial: Vec<f64>,
    conditioned: bool,
}
impl Model {
    fn parse(&self, value: &Value, op: KernelOperation) -> Result<Vec<f64>, KernelBindingError> {
        let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
        let p = value
            .as_object()
            .ok_or_else(|| invalid("parameters must be an object"))?;
        if p.len() != self.parameters.len() {
            return Err(invalid(
                "composite parameters must match the complete suffixed parameter set",
            ));
        }
        self.parameters.iter().map(|spec|{
            let value=required_f64(p,&spec.name,op)?;
            if !value.is_finite() || (spec.positive && value<0.) || (spec.capacity.is_some() && value<=0.){return Err(invalid("growth coefficients must be nonnegative, capacities positive, interactions/midpoints finite signed values"));}Ok(value)
        }).collect()
    }
    fn value(&self, p: &[f64]) -> Value {
        Value::Object(
            self.parameters
                .iter()
                .zip(p)
                .map(|(s, v)| (s.name.clone(), json!(v)))
                .collect(),
        )
    }
    fn derivative(&self, y: &[f64], p: &[f64]) -> Vec<f64> {
        let n = self.options.components.len();
        self.options
            .components
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let a = &p[self.starts[i]..];
                let v = y[i];
                let growth = match model.as_str() {
                    "bass" => (a[0] + a[1] * v / a[2]) * (a[2] - v),
                    "logistic" => a[1] * v * (1. - v / a[0]),
                    "gompertz" => {
                        if v <= 0. {
                            0.
                        } else {
                            a[2] * v * (a[0].ln() - v.ln())
                        }
                    }
                    "fisher_pry" => a[0] * v * (1. - v),
                    _ => unreachable!(),
                };
                let interaction = (0..n)
                    .filter(|j| *j != i)
                    .enumerate()
                    .map(|(offset, j)| p[self.base_count + i * (n - 1) + offset] * y[j])
                    .sum::<f64>();
                growth - interaction
            })
            .collect()
    }
    fn predict(&self, time: &[f64], p: &[f64]) -> Result<Vec<Vec<f64>>, String> {
        integrate(time, self.origin, &self.initial, |_, y| {
            self.derivative(y, p)
        })
    }
    fn prediction(&self, time: &[f64], p: &[f64], rows: &[Vec<f64>]) -> Result<Value, String> {
        let rates: Vec<_> = rows.iter().map(|y| self.derivative(y, p)).collect();
        if rows.iter().chain(&rates).flatten().any(|v| !v.is_finite()) {
            return Err("composite levels/rates exceeded finite numerical domain".into());
        }
        let unused: Vec<_> = self
            .parameters
            .iter()
            .filter(|s| s.unused)
            .map(|s| s.name.clone())
            .collect();
        let warnings: Vec<_> = if rows.iter().flatten().any(|v| *v < 0.) {
            vec!["signed composite interactions produced negative levels; inspect assumptions"]
        } else {
            vec![]
        };
        Ok(
            json!({"rows":rows,"rates":rates,"shape":[time.len(),self.options.components.len()],"dtype":"float64","metadata":{"origin_mode":self.options.origin_mode,"initial_time":self.origin,"initial_values":self.initial,"components":self.options.components,"rate_semantics":"signed composite ODE RHS, not incidence or normalized shares","unused_parameters":unused,"warnings":warnings,"solver":"adaptive step-doubling RK4 with Richardson correction; absolute/relative tolerances 1e-10 and 200000-step budget","simulation_semantics":"deterministic ODE path; stochastic draws unavailable"}}),
        )
    }
}
pub(crate) fn execute(r: &KernelRequest) -> Result<KernelResponse, KernelBindingError> {
    let op = r.operation;
    let invalid = |s: &str| KernelBindingError::invalid_request(op, s);
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
        return Err(invalid("unknown composite payload option"));
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
        .is_some_and(|m| m != "composite")
    {
        return Err(invalid("state model_key mismatch"));
    }
    let mut o: Options = serde_json::from_value(
        p.get("constructor_kwargs")
            .or_else(|| p.get("model_kwargs"))
            .or_else(|| p.get("state").and_then(|s| s.get("constructor_kwargs")))
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|e| invalid(&e.to_string()))?;
    let n = o.components.len();
    if n == 0
        || n > 16
        || o.components.iter().any(|c| names(c).is_none())
        || !["positive", "historical_observed"].contains(&o.capacity_bounds.as_str())
        || !["fixed", "historical_batch"].contains(&o.origin_mode.as_str())
    {
        return Err(invalid(
            "composite requires 1..=16 supported scalar components and fixed/historical_batch origin mode",
        ));
    }
    if o.alpha.as_ref().is_some_and(|alpha| {
        alpha.len() != n
            || alpha.iter().enumerate().any(|(i, row)| {
                row.len() != n
                    || row
                        .iter()
                        .enumerate()
                        .any(|(j, v)| !v.is_finite() || (i == j && *v != 0.))
            })
    }) {
        return Err(invalid(
            "alpha seed must be a finite square matrix with zero diagonal",
        ));
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
    if input.contains_key("covariates") {
        return Err(invalid("composite base covariates not ported"));
    }
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
    let observed = if op == KernelOperation::FitModel
        || op == KernelOperation::DiagnoseModel
        || input.contains_key("observed")
        || input.contains_key("y")
    {
        Some(crate::substitution::observations(input, n, time.len(), op)?)
    } else {
        None
    };
    let conditioned =
        op == KernelOperation::FitModel && o.origin_mode == "fixed" && o.initial_values.is_none();
    let (origin, initial) = if o.origin_mode == "historical_batch" {
        if o.initial_time.is_some() || o.initial_values.is_some() {
            return Err(invalid(
                "historical_batch resets zero state at each batch start; omit initial_time/initial_values",
            ));
        }
        (time.first().copied().unwrap_or(0.), vec![0.; n])
    } else if conditioned {
        let origin = o.initial_time.unwrap_or(time[0]);
        if origin != time[0] {
            return Err(invalid(
                "inferred initial state requires initial_time equal first observation time",
            ));
        }
        let initial = observed.as_ref().unwrap()[0].clone();
        o.initial_time = Some(origin);
        o.initial_values = Some(initial.clone());
        (origin, initial)
    } else {
        let origin = o.initial_time.unwrap_or(0.);
        let initial = o.initial_values.clone().unwrap_or_else(|| vec![0.; n]);
        o.initial_time = Some(origin);
        o.initial_values = Some(initial.clone());
        (origin, initial)
    };
    if !origin.is_finite()
        || origin < 0.
        || time.first().is_some_and(|t| origin > *t)
        || initial.len() != n
        || initial.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(invalid(
            "initial time/state must be finite, nonnegative, aligned and no later than sample times",
        ));
    }
    let mut specifications = vec![];
    let mut starts = vec![];
    for (i, c) in o.components.iter().enumerate() {
        starts.push(specifications.len());
        for name in names(c).unwrap() {
            specifications.push(Parameter {
                name: format!("{name}_{}", i + 1),
                positive: !["x0", "t0"].contains(name),
                unused: ["x0", "t0", "b"].contains(name),
                capacity: if ["m", "L", "a"].contains(name) {
                    Some(i)
                } else {
                    None
                },
            });
        }
    }
    let base_count = specifications.len();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                specifications.push(Parameter {
                    name: format!("alpha_{}_{}", i + 1, j + 1),
                    positive: false,
                    unused: false,
                    capacity: None,
                });
            }
        }
    }
    let model = Model {
        options: o,
        parameters: specifications,
        starts,
        base_count,
        origin,
        initial,
        conditioned,
    };
    let offset = usize::from(conditioned);
    let mut quality = None;
    let params = if op == KernelOperation::FitModel {
        let y = observed.as_ref().unwrap();
        if time.len() < 4 + offset {
            return Err(invalid(
                "composite fitting requires four observations after conditioning",
            ));
        }
        let ys = y.iter().flatten().copied().fold(0., f64::max);
        if ys <= 0. {
            return Err(invalid(
                "all-zero observations cannot identify composite growth parameters",
            ));
        }
        let ts = time.iter().map(|t| (*t - origin).abs()).fold(1., f64::max);
        let floors: Vec<_> = model
            .parameters
            .iter()
            .map(|s| {
                s.capacity
                    .filter(|_| model.options.capacity_bounds == "historical_observed")
                    .map(|i| y.iter().map(|r| r[i]).fold(0., f64::max))
                    .unwrap_or(0.)
            })
            .collect();
        let scales: Vec<_> = model
            .parameters
            .iter()
            .map(|s| if s.capacity.is_some() { ys } else { 1. / ts })
            .collect();
        let (initial, limit) = crate::fitting::options(r)?;
        let seed = if let Some(initial) = initial {
            let p = model.parse(&initial, op)?;
            if p.iter()
                .zip(&floors)
                .zip(&model.parameters)
                .any(|((p, f), s)| s.capacity.is_some() && p < f)
            {
                return Err(invalid(
                    "initial capacities must be at least their maximum observed series value",
                ));
            }
            p
        } else {
            let mut seed = vec![];
            for (i, c) in model.options.components.iter().enumerate() {
                for name in names(c).unwrap() {
                    seed.push(match *name {
                        "m" | "L" | "a" => {
                            y.iter().map(|r| r[i]).fold(0., f64::max).max(ys * 0.1) * 1.2
                        }
                        "p" => 0.03 / ts,
                        "q" => 0.35 / ts,
                        "b" => 1.,
                        "x0" | "t0" => (time[0] + time.last().unwrap()) / 2.,
                        _ => 1. / ts,
                    });
                }
            }
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        seed.push(model.options.alpha.as_ref().map(|a| a[i][j]).unwrap_or(0.));
                    }
                }
            }
            seed
        };
        let active: Vec<_> = model
            .parameters
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.unused)
            .map(|(i, _)| i)
            .collect();
        let transform = |p: &[f64]| {
            active
                .iter()
                .map(|i| {
                    if model.parameters[*i].positive {
                        ((p[*i] - floors[*i]) / scales[*i]).max(1e-12).ln()
                    } else {
                        p[*i] / scales[*i]
                    }
                })
                .collect::<Vec<_>>()
        };
        let untransform = |theta: &[f64]| {
            let mut p = seed.clone();
            for (i, t) in active.iter().zip(theta) {
                p[*i] = if model.parameters[*i].positive {
                    floors[*i] + t.exp() * scales[*i]
                } else {
                    t * scales[*i]
                };
            }
            p
        };
        let theta = transform(&seed);
        let starts = vec![
            theta.clone(),
            theta.iter().map(|v| v + 0.1).collect(),
            theta.iter().map(|v| v - 0.1).collect(),
        ];
        let target: Vec<_> = y[offset..].iter().flatten().map(|v| v / ys).collect();
        let fit = least_squares(
            &target,
            &starts,
            |theta| {
                model
                    .predict(&time, &untransform(theta))
                    .map(|rows| rows[offset..].iter().flatten().map(|v| v / ys).collect())
            },
            limit,
        )
        .map_err(|e| invalid(&e))?;
        let parameters = untransform(&fit.parameters);
        quality = Some((fit.report, fit.identifiability));
        parameters
    } else {
        model.parse(
            p.get("state")
                .and_then(|s| s.get("parameters"))
                .or_else(|| p.get("parameters"))
                .ok_or_else(|| invalid("parameters required"))?,
            op,
        )?
    };
    let eval = if time.is_empty() {
        vec![origin]
    } else {
        time.clone()
    };
    let rows = model.predict(&eval, &params).map_err(|e| invalid(&e))?;
    let predictions = model
        .prediction(&eval, &params, &rows)
        .map_err(|e| invalid(&e))?;
    let state = json!({"model_key":"composite","parameters":model.value(&params),"constructor_kwargs":model.options,"predict_kwargs":{}});
    let absent: Vec<_> = model
        .parameters
        .iter()
        .filter(|s| s.unused)
        .map(|s| s.name.clone())
        .collect();
    let estimated: Vec<_> = model
        .parameters
        .iter()
        .filter(|s| !s.unused)
        .map(|s| s.name.clone())
        .collect();
    let result = if matches!(
        op,
        KernelOperation::PredictModel | KernelOperation::SimulateModel
    ) {
        predictions
    } else {
        let mut out = json!({"model_key":"composite","model_name":"CompositeDiffusionModel","parameters":model.value(&params),"state":state,"predictions":predictions,"details":{"components":model.options.components,"origin_mode":model.options.origin_mode,"initial_time":origin,"initial_values":model.initial,"absent_from_rhs":absent}});
        if time.is_empty() {
            out.as_object_mut().unwrap().remove("predictions");
        }
        if let Some(y) = observed {
            let rss = y[offset..]
                .iter()
                .flatten()
                .zip(rows[offset..].iter().flatten())
                .map(|(y, p)| (y - p).powi(2))
                .sum::<f64>();
            if !rss.is_finite() {
                return Err(invalid("composite residual sum of squares overflow"));
            }
            let mut d = crate::substitution::diagnostics(
                &time[offset..],
                &y[offset..],
                &rows[offset..],
                "CompositeDiffusionModel",
                estimated.len(),
            );
            d["initial_condition"] = json!({"source":if model.conditioned{"first_observed_row"}else{"explicit_stored_or_zero"},"excluded_conditioning_rows":offset,"uncertainty_propagated":false});
            d["structural_identifiability"] = json!({"absent_from_rhs":absent,"estimated_parameters":estimated,"absent_parameters_held_fixed":true});
            d["metrics"]["AIC"] = Value::Null;
            d["metrics"]["BIC"] = Value::Null;
            d["information_criteria"] = json!({"available":false,"reason":"conditional initial state and composite structural degrees of freedom require qualification"});
            d["warnings"].as_array_mut().unwrap().push(json!("Logistic x0, Gompertz b and Fisher-Pry t0 do not enter the ODE and are held fixed; local rank covers only active parameters. Signed interactions may produce negative or nonmonotone levels."));
            if let Some((optimizer, mut identifiability)) = quality {
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
                        .push(json!("active ODE parameters are locally rank deficient"));
                }
                identifiability["scope"] = json!(
                    "active ODE parameters only; not full-model uniqueness or calibrated uncertainty"
                );
                d["optimizer"] = optimizer;
                d["identifiability"] = identifiability;
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
        metadata: json!({"runtime":"rust_native","family":"composite"}),
    })
}
