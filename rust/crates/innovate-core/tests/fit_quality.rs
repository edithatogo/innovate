use innovate_core::{KernelBinding, json};

#[test]
fn scalar_native_models_report_actual_optimizer_termination_and_local_rank() {
    let time: Vec<f64> = (0..25).map(f64::from).collect();
    for (model, parameters, count) in [
        ("bass", json!({"p":0.03,"q":0.35,"m":100.}), 3),
        ("logistic", json!({"L":100.,"k":0.6,"x0":8.}), 3),
        ("gompertz", json!({"a":100.,"b":3.,"c":0.3}), 3),
        ("fisher_pry", json!({"alpha":0.6,"t0":8.}), 2),
    ] {
        let binding = KernelBinding::new();
        let predicted = binding
            .predict_model(
                model,
                json!({"inputs":{"time":time},"parameters":parameters}),
            )
            .unwrap()
            .result
            .unwrap();
        let fit = binding
            .fit_model(
                model,
                json!({"inputs":{"time":time,"observed":predicted["values"]}}),
            )
            .unwrap()
            .result
            .unwrap();
        assert_eq!(
            fit["diagnostics"]["optimizer"]["converged"], true,
            "{model}: {fit}"
        );
        let optimizer = &fit["diagnostics"]["optimizer"];
        let selected = optimizer["selected_start_index"].as_u64().unwrap() as usize;
        let start = &optimizer["start_reports"][selected];
        assert_eq!(start["scaled_rss"], optimizer["scaled_rss"]);
        assert!(
            start["initial_scaled_rss"].as_f64().unwrap() >= start["scaled_rss"].as_f64().unwrap()
        );
        for key in [
            "initial_transformed_parameters",
            "final_transformed_parameters",
        ] {
            assert_eq!(start[key].as_array().unwrap().len(), count);
        }
        assert_eq!(fit["diagnostics"]["identifiability"]["rank"], count);
        assert_eq!(
            fit["diagnostics"]["uncertainty"]["report_type"],
            "point_estimate"
        );
    }
}

#[test]
fn custom_initial_parameters_and_iteration_limits_are_honored() {
    let time: Vec<f64> = (0..20).map(f64::from).collect();
    let observed: Vec<_> = time
        .iter()
        .map(|t| 100. / (1. + (-0.6 * (t - 8.)).exp()))
        .collect();
    let fit = KernelBinding::new()
        .fit_model(
            "logistic",
            json!({"inputs":{"time":time,"observed":observed},
        "fit_options":{"max_iterations":1,"initial_parameters":{"L":250.,"k":0.1,"x0":2.}}}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], false);
    assert_eq!(fit["diagnostics"]["optimizer"]["max_iterations"], 1);
    assert!(
        !fit["diagnostics"]["warnings"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn custom_seeds_cannot_bypass_option_or_observation_validation() {
    let base = json!({"inputs":{"time":[0.,1.,2.,3.],"observed":[1.,2.,3.,4.]},"fit_options":{"initial_parameters":{"L":10.,"k":0.4,"x0":2.}}});
    for (key, value) in [
        ("fitter_options", json!({"loss":"ignored"})),
        ("constructor_kwargs", json!({"unknown":true})),
        ("constructor_kwargs", json!(false)),
        ("predict_kwargs", json!({"unknown":true})),
    ] {
        let mut p = base.clone();
        p[key] = value;
        assert!(KernelBinding::new().fit_model("logistic", p).is_err());
    }
    for options in [
        json!({"max_iterations":0}),
        json!({"max_iterations":2001}),
        json!({"loss":"ignored"}),
        json!([]),
        json!({"initial_parameters":{"L":1.,"k":-1.,"x0":0.}}),
    ] {
        let mut p = base.clone();
        p["fit_options"] = options;
        assert!(KernelBinding::new().fit_model("logistic", p).is_err());
    }
    let mut p = base;
    p["inputs"]["covariates"] = json!({"price":[1.,2.,3.,4.]});
    assert!(KernelBinding::new().fit_model("logistic", p).is_err());
}

#[test]
fn excessive_initial_parameter_scale_is_a_structured_error_not_false_convergence() {
    let e=KernelBinding::new().fit_model("logistic",json!({"inputs":{"time":[0.,1.,2.,3.],"observed":vec![1e-12;4]},"fit_options":{"initial_parameters":{"L":1e140,"k":1e15,"x0":1.}}})).unwrap_err();
    assert_eq!(e.code, "invalid_request");
    assert!(e.message.contains("finite numerical domain"));
}

#[test]
fn noisy_logistic_fit_recovers_the_held_out_trajectory() {
    assert_held_out_trajectory("logistic", |t| 120. / (1. + (-0.38 * (t - 12.)).exp()));
}

#[test]
fn noisy_gompertz_fit_recovers_the_held_out_trajectory() {
    assert_held_out_trajectory("gompertz", |t| 120. * (-4. * (-0.16 * t).exp()).exp());
}

#[test]
fn noisy_bass_fit_recovers_the_held_out_trajectory() {
    assert_held_out_trajectory("bass", |t| {
        let decay = (-0.24 * t).exp();
        120. * (1. - decay) / (1. + 7. * decay)
    });
}

#[test]
fn noisy_fisher_pry_fit_recovers_the_held_out_share() {
    assert_held_out_trajectory("fisher_pry", |t| 1. / (1. + (-0.3 * (t - 12.)).exp()));
}

fn assert_held_out_trajectory(model: &str, truth: impl Fn(f64) -> f64) {
    let time: Vec<_> = (0..31).map(f64::from).collect();
    let (rank, tolerance) = if model == "fisher_pry" {
        (2, 0.01)
    } else {
        (3, 1.)
    };
    let errors = [0.8, -1.1, 0.4, -0.6, 1.3, -0.3];
    let observed: Vec<_> = time[..22]
        .iter()
        .enumerate()
        .map(|(i, &t)| truth(t) + tolerance * errors[i % errors.len()])
        .collect();
    let binding = KernelBinding::new();
    let fit = binding
        .fit_model(
            model,
            json!({"inputs":{"time":time[..22],"observed":observed}}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
    assert_eq!(fit["diagnostics"]["identifiability"]["rank"], rank);
    let prediction = binding
        .predict_model(
            model,
            json!({"inputs":{"time":time[22..]},"state":fit["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    let predictions = prediction["values"].as_array().unwrap();
    assert_eq!(predictions.len(), 9);
    for (&t, value) in time[22..].iter().zip(predictions) {
        assert!(
            (value.as_f64().unwrap() - truth(t)).abs() < tolerance,
            "{t}: {value}"
        );
    }
}

#[test]
fn saturated_logistic_fit_reports_unidentified_rate_and_midpoint() {
    let time: Vec<_> = (0..8).collect();
    let fit = KernelBinding::new()
        .fit_model(
            "logistic",
            json!({
                "inputs":{"time":time,"observed":vec![100.;8]},
                "fit_options":{"initial_parameters":{"L":100.,"k":1.,"x0":-100.}}
            }),
        )
        .unwrap()
        .result
        .unwrap();
    let diagnostics = &fit["diagnostics"];
    let sensitivity = &diagnostics["identifiability"];
    assert_eq!(sensitivity["status"], "rank_deficient");
    assert_eq!(sensitivity["rank"], 1);
    assert_eq!(sensitivity["parameters"], 3);
    assert_eq!(
        sensitivity["excluded_sensitivity_column_indices"],
        json!([1, 2])
    );
    assert_eq!(diagnostics["uncertainty"]["report_type"], "point_estimate");
}
