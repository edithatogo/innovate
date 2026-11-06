//! Native competitive diffusion equations preserved from the pre-migration models.
use crate::numerics::{integrate, least_squares};
use crate::validation::{numeric_array, numeric_array_from_aliases, object_section, required_f64};
use crate::{
    KERNEL_SCHEMA_VERSION, KernelBindingError, KernelOperation, KernelRequest, KernelResponse,
};
use serde_json::{Map, Value, json};

struct Model {
    key: String,
    n: usize,
    names: Vec<String>,
    origin: f64,
    initial: Vec<f64>,
    conditioned_initial: bool,
}
impl Model {
    fn name(&self) -> &str {
        if self.key == "lotka_volterra" {
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
        self.names.iter().map(|name| {
            let x = required_f64(p,name,op)?;
            if !x.is_finite() || x < 0. || (name.starts_with('m') && x == 0.) {
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
    fn derivative(&self, y: &[f64], p: &[f64]) -> Vec<f64> {
        if self.key == "lotka_volterra" {
            return vec![
                p[0] * y[0] * (1. - y[0]) - p[1] * y[0] * y[1],
                p[2] * y[1] * (1. - y[1]) - p[3] * y[0] * y[1],
            ];
        }
        (0..self.n)
            .map(|i| {
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
        integrate(time, self.origin, &self.initial, |_, y| {
            self.derivative(y, p)
        })
    }
    fn prediction(&self, time: &[f64], p: &[f64], rows: &[Vec<f64>]) -> Result<Value, String> {
        let rates: Vec<_> = rows.iter().map(|y| self.derivative(y, p)).collect();
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
            "rate_semantics":"signed derivative of each level; not gross incidence","warnings":warnings,
            "normalization":"separate product potentials; rows are not constrained to sum to one"}}),
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
    if kw.iter().any(|(k, v)| {
        !(["n_products", "initial_time", "initial_values"].contains(&k.as_str())
            || (k == "covariates" && v.as_array().is_some_and(Vec::is_empty)))
    }) || inputs.contains_key("covariates")
    {
        return Err(KernelBindingError::unsupported_native_operation(
            op,
            "competitive covariates, events and custom options remain migration obligations",
        ));
    }
    if key == "lotka_volterra" && kw.contains_key("n_products") {
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
    let n = if key == "lotka_volterra" {
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
    let names = if key == "lotka_volterra" {
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
    Ok(Model {
        key,
        n,
        names,
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
        let scales: Vec<_> = model
            .names
            .iter()
            .map(|name| {
                if name.starts_with('m') {
                    yscale
                } else if name.starts_with("alpha_") {
                    1.
                } else {
                    1. / tscale
                }
            })
            .collect();
        let transform = |p: Vec<f64>| {
            p.iter()
                .zip(&scales)
                .map(|(p, s)| (p / s).max(1e-12).ln())
                .collect::<Vec<_>>()
        };
        let untransform = |theta: &[f64]| {
            theta
                .iter()
                .zip(&scales)
                .map(|(t, s)| t.exp() * s)
                .collect::<Vec<_>>()
        };
        let (initial, limit) = crate::fitting::options(request)?;
        let mut starts = Vec::new();
        if let Some(initial) = initial {
            starts.push(transform(model.parameters(&initial, op)?));
        }
        for rate in [0.5, 2., 5.] {
            let p = model
                .names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if name.starts_with('m') {
                        observed
                            .iter()
                            .map(|r| r[i - 2 * model.n])
                            .fold(0., f64::max)
                            .max(yscale * 0.1)
                            * 1.2
                    } else if name.starts_with("alpha_") {
                        0.1
                    } else if name.starts_with('p') || name.starts_with("beta") {
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
