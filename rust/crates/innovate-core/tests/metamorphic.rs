//! Bounded metamorphic contracts for native point predictions, not model validity.
//!
//! Eight fixed curves, six time points, three nonidentity scale factors and six
//! mixture permutations are the entire corpus. No wall clock, random generator,
//! network, optimizer or unbounded retry participates in replay. Covariates,
//! event splits, fitted-parameter covariance and stochastic simulation are not
//! covered by these closed-form transformations. Kairos replay has its own tests.
use innovate_core::{KernelBinding, KernelRequest, json};
use serde_json::{Map, Value};

const TIME: [f64; 6] = [0., 0.125, 1.25, 4., 11., 29.];
// Well-conditioned finite fixtures: permit normal floating-point evaluation
// order differences, not a fit error or a percentage-level scientific tolerance.
const ABS_TOL: f64 = 1e-12;
const REL_TOL: f64 = 1e-12;

struct Curve {
    model: &'static str,
    parameters: Value,
    rates: &'static [&'static str],
    location: Option<&'static str>,
    amplitude: Option<&'static str>,
    // Analytic landmarks independently anchor levels and derivatives, so a
    // constant/zero implementation cannot pass solely through self-comparison.
    landmark: (usize, f64, f64),
}

fn curves() -> Vec<Curve> {
    let mut cases = Vec::new();
    for (index, capacity, rate) in [(3, 120., 0.45), (4, 3.2, 0.06)] {
        cases.push(Curve {
            model: "logistic",
            parameters: json!({"L":capacity,"k":rate,"x0":TIME[index]}),
            rates: &["k"],
            location: Some("x0"),
            amplitude: Some("L"),
            landmark: (index, capacity / 2., capacity * rate / 4.),
        });
        cases.push(Curve {
            model: "fisher_pry",
            parameters: json!({"alpha":rate,"t0":TIME[index]}),
            rates: &["alpha"],
            location: Some("t0"),
            amplitude: None, // A probability has no free amplitude parameter.
            landmark: (index, 0.5, rate / 4.),
        });
        let c = 1. / TIME[index];
        cases.push(Curve {
            model: "gompertz",
            parameters: json!({"a":capacity,"b":std::f64::consts::E,"c":c}),
            rates: &["c"],
            location: None,
            amplitude: Some("a"),
            landmark: (
                index,
                capacity / std::f64::consts::E,
                capacity * c / std::f64::consts::E,
            ),
        });
    }
    for (p, q, m) in [(0.03, 0.35, 100.), (0.2, 0., 7.)] {
        cases.push(Curve {
            model: "bass",
            parameters: json!({"p":p,"q":q,"m":m}),
            rates: &["p", "q"],
            location: None,
            amplitude: Some("m"),
            landmark: (0, 0., m * p),
        });
    }
    cases
}

fn close(actual: f64, expected: f64, context: &str) {
    let tolerance = ABS_TOL + REL_TOL * actual.abs().max(expected.abs());
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= tolerance,
        "{context}: actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.17e}"
    );
}

fn number(value: &Value, context: &str) -> f64 {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .unwrap_or_else(|| panic!("{context}: expected finite number, received {value}"))
}

fn invoke(request: KernelRequest, context: &str) -> Value {
    let response = KernelBinding::new()
        .invoke(&request)
        .unwrap_or_else(|error| panic!("{context}: {error}; request={request:?}"));
    assert_eq!(response.metadata["runtime"], "rust_native", "{context}");
    assert!(response.error.is_none(), "{context}: {:?}", response.error);
    response
        .result
        .unwrap_or_else(|| panic!("{context}: result missing"))
}

fn prediction(
    case: &Curve,
    time: &[f64],
    parameters: &Value,
    simulate: bool,
    context: &str,
) -> (Vec<f64>, Vec<f64>) {
    let payload = json!({"inputs":{"time":time},"parameters":parameters});
    let request = if simulate {
        KernelRequest::simulate_model(case.model, payload)
    } else {
        KernelRequest::predict_model(case.model, payload)
    };
    let result = invoke(request, context);
    assert_eq!(result["shape"], json!([time.len()]), "{context}");
    let values = result["values"].as_array().expect("values array");
    let rates = result["rates"].as_array().expect("rates array");
    assert_eq!(values.len(), time.len(), "{context}: value count");
    assert_eq!(rates.len(), time.len(), "{context}: rate count");
    for row in rates {
        assert_eq!(row.as_array().map(Vec::len), Some(1), "{context}: rate row");
    }
    (
        values.iter().map(|v| number(v, context)).collect(),
        rates.iter().map(|v| number(&v[0], context)).collect(),
    )
}

fn baseline(case: &Curve, simulate: bool, context: &str) -> (Vec<f64>, Vec<f64>) {
    let (values, rates) = prediction(case, &TIME, &case.parameters, simulate, context);
    let (index, level, rate) = case.landmark;
    close(values[index], level, &format!("{context}: analytic level"));
    close(
        rates[index],
        rate,
        &format!("{context}: analytic derivative"),
    );
    assert!(values[5] - values[0] > 0.1, "{context}: nonconstant curve");
    assert!(rates.iter().any(|v| *v > 0.01), "{context}: positive rate");
    (values, rates)
}

#[test]
fn time_unit_covariance_preserves_levels_and_rescales_rates() {
    for (case_id, case) in curves().iter().enumerate() {
        for simulate in [false, true] {
            let context = format!("case={case_id}, model={}, simulate={simulate}", case.model);
            let (values, rates) = baseline(case, simulate, &context);
            // t' = c*t, inverse-time parameters /= c, locations *= c.
            // Levels stay unchanged; dy/dt' = (dy/dt)/c. No origin shift.
            for scale in [1. / 365., 7., 365.] {
                let context = format!("{context}, time_scale={scale:.17e}");
                let time: Vec<_> = TIME.iter().map(|t| t * scale).collect();
                let mut parameters = case.parameters.clone();
                for &key in case.rates {
                    parameters[key] = json!(number(&case.parameters[key], &context) / scale);
                }
                if let Some(key) = case.location {
                    parameters[key] = json!(number(&case.parameters[key], &context) * scale);
                }
                let (actual, actual_rates) =
                    prediction(case, &time, &parameters, simulate, &context);
                for index in 0..TIME.len() {
                    close(
                        actual[index],
                        values[index],
                        &format!("{context}, level[{index}]"),
                    );
                    close(
                        actual_rates[index],
                        rates[index] / scale,
                        &format!("{context}, rate[{index}]"),
                    );
                }
            }
        }
    }
}

#[test]
fn amplitude_covariance_rescales_levels_and_rates() {
    for (case_id, case) in curves().iter().enumerate() {
        let Some(key) = case.amplitude else { continue };
        for simulate in [false, true] {
            let context = format!("case={case_id}, model={}, simulate={simulate}", case.model);
            let (values, rates) = baseline(case, simulate, &context);
            for scale in [0.001, 7., 1000.] {
                let context = format!("{context}, amplitude_scale={scale:.17e}");
                let mut parameters = case.parameters.clone();
                parameters[key] = json!(number(&case.parameters[key], &context) * scale);
                let (actual, actual_rates) =
                    prediction(case, &TIME, &parameters, simulate, &context);
                for index in 0..TIME.len() {
                    close(
                        actual[index],
                        values[index] * scale,
                        &format!("{context}, level[{index}]"),
                    );
                    close(
                        actual_rates[index],
                        rates[index] * scale,
                        &format!("{context}, rate[{index}]"),
                    );
                }
            }
        }
    }
}

fn mixture_payload(permutation: &[usize; 3]) -> Value {
    let components = [
        ("logistic", json!({"L":120.,"k":0.45,"x0":4.}), 0.2),
        ("gompertz", json!({"a":180.,"b":2.2,"c":0.3}), 0.3),
        ("bass", json!({"p":0.08,"q":0.24,"m":100.}), 0.5),
    ];
    let mut parameters = Map::new();
    let mut models = Vec::new();
    for (target, &source) in permutation.iter().enumerate() {
        let (model, component, weight) = &components[source];
        models.push(*model);
        for (key, value) in component.as_object().unwrap() {
            parameters.insert(format!("model_{target}_{key}"), value.clone());
        }
        parameters.insert(format!("weight_{target}"), json!(weight));
    }
    parameters.insert("noise_scale".into(), json!(30.));
    json!({
        "inputs":{"time":TIME,"observed":[10.,20.,40.,65.,110.,130.]},
        "constructor_kwargs":{"components":models},"parameters":parameters
    })
}

#[test]
fn mixture_component_permutation_preserves_aggregates_and_permutes_responsibilities() {
    let baseline = invoke(
        KernelRequest::diagnose_model("mixture", mixture_payload(&[0, 1, 2])),
        "mixture baseline",
    );
    let rows = baseline["predictions"]["component_rows"]
        .as_array()
        .unwrap();
    assert_eq!(rows.len(), TIME.len());
    // Distinct components and weights prevent permutation equality being
    // satisfied by an accidentally duplicated component or uniform fixture.
    let first: Vec<_> = (0..3)
        .map(|i| number(&rows[0][i], "component control"))
        .collect();
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        assert!((first[i] - first[j]).abs() > 1.);
    }
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(row.as_array().map(Vec::len), Some(3));
        let weighted: f64 = [0.2, 0.3, 0.5]
            .iter()
            .enumerate()
            .map(|(i, weight)| weight * number(&row[i], "weighted control"))
            .sum();
        close(
            number(
                &baseline["predictions"]["values"][index],
                "weighted control",
            ),
            weighted,
            &format!("mixture baseline, weighted level[{index}]"),
        );
    }
    let responsibilities = baseline["responsibilities"].as_array().unwrap();
    assert_eq!(responsibilities.len(), TIME.len());
    let mut saw_nonuniform = false;
    for (index, row) in responsibilities.iter().enumerate() {
        assert_eq!(row.as_array().map(Vec::len), Some(3));
        let values: Vec<_> = (0..3)
            .map(|i| number(&row[i], "responsibility control"))
            .collect();
        assert!(values.iter().all(|v| (0.0..=1.0).contains(v)));
        close(
            values.iter().sum(),
            1.,
            &format!("mixture baseline, responsibility sum[{index}]"),
        );
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        saw_nonuniform |= high - low > 0.05;
    }
    assert!(
        saw_nonuniform,
        "fixture must exercise unequal responsibilities"
    );
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let context = format!("mixture permutation={permutation:?}");
        let result = invoke(
            KernelRequest::diagnose_model("mixture", mixture_payload(&permutation)),
            &context,
        );
        assert_eq!(
            result["predictions"]["values"].as_array().map(Vec::len),
            Some(TIME.len()),
            "{context}"
        );
        assert_eq!(
            result["predictions"]["component_rows"]
                .as_array()
                .map(Vec::len),
            Some(TIME.len()),
            "{context}"
        );
        assert_eq!(
            result["responsibilities"].as_array().map(Vec::len),
            Some(TIME.len()),
            "{context}"
        );
        close(
            number(&result["objective"], &context),
            number(&baseline["objective"], &context),
            &format!("{context}, log likelihood"),
        );
        for index in 0..TIME.len() {
            assert_eq!(
                result["predictions"]["component_rows"][index]
                    .as_array()
                    .map(Vec::len),
                Some(3),
                "{context}: component count"
            );
            assert_eq!(
                result["responsibilities"][index].as_array().map(Vec::len),
                Some(3),
                "{context}: responsibility count"
            );
            close(
                number(&result["predictions"]["values"][index], &context),
                number(&baseline["predictions"]["values"][index], &context),
                &format!("{context}, level[{index}]"),
            );
            for (target, &source) in permutation.iter().enumerate() {
                for path in ["component_rows", "responsibilities"] {
                    let (actual, expected) = if path == "component_rows" {
                        (
                            &result["predictions"][path][index][target],
                            &baseline["predictions"][path][index][source],
                        )
                    } else {
                        (&result[path][index][target], &baseline[path][index][source])
                    };
                    close(
                        number(actual, &context),
                        number(expected, &context),
                        &format!("{context}, {path}[{index}][{target}] -> [{source}]"),
                    );
                }
            }
        }
    }
}
