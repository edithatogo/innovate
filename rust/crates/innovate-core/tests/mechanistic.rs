use innovate_core::{KernelBinding, json};
#[test]
fn bass_linear_innovation_covariate_and_saved_history_match_exact_solution() {
    let p = json!({"inputs":{"time":[0.,1.,2.,4.],"covariates":{"dose":[0.,1.,2.,4.]}},"constructor_kwargs":{"covariates":["dose"]},"parameters":{"p":0.1,"q":0.,"m":100.,"beta_p_dose":0.02,"beta_q_dose":0.,"beta_m_dose":0.}});
    let s = KernelBinding::new()
        .summarize_model("bass", p)
        .unwrap()
        .result
        .unwrap();
    for (i, t) in [0_f64, 1., 2., 4.].iter().enumerate() {
        let expected = 100. * (1. - (-0.1 * t - 0.01 * t * t).exp());
        assert!((s["predictions"]["values"][i].as_f64().unwrap() - expected).abs() < 1e-7);
    }
    let a = KernelBinding::new()
        .predict_model(
            "bass",
            json!({"inputs":{"time":[5.,6.],"covariates":{"dose":[5.,6.]}},"state":s["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    assert_eq!(
        a["state"]["constructor_kwargs"]["covariate_history"]["time"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
    let expected = 100. * (1. - (-0.96_f64).exp());
    assert!((a["values"][1].as_f64().unwrap() - expected).abs() < 1e-7);
    assert!((a["rates"][1][0].as_f64().unwrap() - 0.22 * (100. - expected)).abs() < 1e-7);
}
#[test]
fn gompertz_event_has_continuous_state_and_right_hand_rate() {
    let a=KernelBinding::new().predict_model("gompertz",json!({"inputs":{"time":[0.,1.,2.,3.,5.]},"constructor_kwargs":{"t_event":2.,"initial_value":1.},"parameters":{"a":100.,"b":77.,"c":0.3,"a_post":200.,"b_post":99.,"c_post":0.2}})).unwrap().result.unwrap();
    let at_event = 100. * ((0.01_f64).ln() * (-0.6_f64).exp()).exp();
    assert!((a["values"][2].as_f64().unwrap() - at_event).abs() < 1e-7);
    let final_value = 200. * ((at_event / 200.).ln() * (-0.6_f64).exp()).exp();
    assert!((a["values"][4].as_f64().unwrap() - final_value).abs() < 1e-7);
    assert!(
        (a["rates"][2][0].as_f64().unwrap() - 0.2 * at_event * (200. / at_event).ln()).abs() < 1e-7
    );
}

#[test]
fn bass_covariate_event_fit_matches_independent_reference_and_retains_history() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/mechanistic_scipy.json")).unwrap();
    let n = f["training_rows"].as_u64().unwrap() as usize;
    let slice = |key: &str, from: usize, to: usize| json!(&f[key].as_array().unwrap()[from..to]);
    let mut seed = f["parameters"].clone();
    seed["beta_p_dose"] = json!(0.004);
    for fit_options in [json!({"initial_parameters":seed}), json!({})] {
        let fit=KernelBinding::new().fit_model("bass",json!({"inputs":{"time":slice("time",0,n),"observed":slice("observed",0,n),"covariates":{"dose":slice("dose",0,n)}},"constructor_kwargs":{"covariates":["dose"],"t_event":10.},"fit_options":fit_options})).unwrap().result.unwrap();
        assert_eq!(fit["diagnostics"]["optimizer"]["converged"], true);
        assert!(fit["diagnostics"]["metrics"]["RSS"].as_f64().unwrap() < 1e-5);
        let end = f["time"].as_array().unwrap().len();
        let a=KernelBinding::new().predict_model("bass",json!({"inputs":{"time":slice("time",n,end),"covariates":{"dose":slice("dose",n,end)}},"state":fit["state"]})).unwrap().result.unwrap();
        assert!(
            (a["values"][end - n - 1].as_f64().unwrap() - f["observed"][end - 1].as_f64().unwrap())
                .abs()
                < 0.001
        );
        assert_eq!(
            fit["state"]["constructor_kwargs"]["covariate_history"]["time"]
                .as_array()
                .unwrap()
                .len(),
            n
        );
    }
}
#[test]
fn explicit_legacy_layout_and_origin_are_not_silent_new_defaults() {
    let p = json!({"inputs":{"time":[0.,1.,2.,3.],"covariates":{"x":[1.,1.,1.,1.]}},"constructor_kwargs":{"covariates":["x"],"t_event":2.,"coefficient_layout":"historical"},"parameters":{"p":0.1,"q":0.,"m":100.,"p_post":0.2,"q_post":0.,"m_post":100.,"beta_p_x":0.03,"beta_q_x":0.,"beta_m_x":0.}});
    let a = KernelBinding::new()
        .predict_model("bass", p)
        .unwrap()
        .result
        .unwrap();
    assert!((a["values"][1].as_f64().unwrap() - 200. * (1. - (-0.3_f64).exp())).abs() < 1e-7);
    let a=KernelBinding::new().predict_model("bass",json!({"inputs":{"time":[5.,6.]},"constructor_kwargs":{"prediction_mode":"ode_historical_batch"},"parameters":{"p":0.03,"q":0.35,"m":100.}})).unwrap().result.unwrap();
    assert_eq!(a["values"][0], 0.);
    let a=KernelBinding::new().predict_model("gompertz",json!({"inputs":{"time":[5.,6.]},"constructor_kwargs":{"prediction_mode":"ode_historical_batch","projection":"cumulative_max"},"parameters":{"a":100.,"b":1.,"c":0.3}})).unwrap().result.unwrap();
    assert_eq!(a["values"][0], 1e-6);
}
#[test]
fn invalid_forcing_and_historical_revisions_fail_explicitly() {
    let base = json!({"inputs":{"time":[0.,1.],"covariates":{"x":[0.,1.]}},"constructor_kwargs":{"covariates":["x"]},"parameters":{"p":0.1,"q":0.,"m":100.,"beta_p_x":0.01,"beta_q_x":0.,"beta_m_x":0.}});
    let s = KernelBinding::new()
        .summarize_model("bass", base.clone())
        .unwrap()
        .result
        .unwrap();
    assert!(
        KernelBinding::new()
            .predict_model(
                "bass",
                json!({"inputs":{"time":[1.],"covariates":{"x":[2.]}},"state":s["state"]})
            )
            .is_err()
    );
    for cov in [
        json!({"x":[0.]}),
        json!({"other":[0.,1.]}),
        json!({"x":[0.,null]}),
    ] {
        let mut p = base.clone();
        p["inputs"]["covariates"] = cov;
        assert!(KernelBinding::new().predict_model("bass", p).is_err());
    }
    let mut p = base.clone();
    p["parameters"]["beta_p_x"] = json!(-1.);
    assert!(KernelBinding::new().predict_model("bass", p).is_err());
    let mut p = base;
    p["inputs"] = json!({"time":[1.,2.],"covariates":{"x":[1.,2.]}});
    assert!(KernelBinding::new().predict_model("bass", p).is_err());
}

#[test]
fn gompertz_linear_rate_covariate_matches_exact_solution_and_ignores_b_terms() {
    let a=KernelBinding::new().predict_model("gompertz",json!({"inputs":{"time":[0.,1.,2.,4.],"covariates":{"dose":[0.,1.,2.,4.]}},"constructor_kwargs":{"covariates":["dose"],"initial_value":1.},"parameters":{"a":100.,"b":77.,"c":0.1,"beta_a_dose":0.,"beta_b_dose":999.,"beta_c_dose":0.02}})).unwrap().result.unwrap();
    for (i, t) in [0_f64, 1., 2., 4.].iter().enumerate() {
        let expected = 100. * ((0.01_f64).ln() * (-0.1 * t - 0.01 * t * t).exp()).exp();
        assert!((a["values"][i].as_f64().unwrap() - expected).abs() < 1e-7);
    }
}
#[test]
fn gompertz_event_fit_holds_absent_b_and_forecasts_from_saved_origin() {
    let t: Vec<_> = (0..17).map(|i| i as f64 * 0.5).collect();
    let boundary = 100. * ((0.01_f64).ln() * (-0.6_f64).exp()).exp();
    let y: Vec<_> = t
        .iter()
        .map(|t| {
            if *t < 2. {
                100. * ((0.01_f64).ln() * (-0.3 * t).exp()).exp()
            } else {
                200. * ((boundary / 200.).ln() * (-0.2 * (t - 2.)).exp()).exp()
            }
        })
        .collect();
    for (fit_options, b, b_post) in [
        (json!({"initial_parameters":{"a":100.,"b":77.,"c":0.3,"a_post":200.,"b_post":99.,"c_post":0.18}}), 77., 99.),
        (json!({}), 1., 1.),
    ] {
        let f=KernelBinding::new().fit_model("gompertz",json!({"inputs":{"time":t,"observed":y},"constructor_kwargs":{"t_event":2.,"initial_value":1.},"fit_options":fit_options})).unwrap().result.unwrap();
        assert_eq!(f["diagnostics"]["optimizer"]["converged"], true);
        assert_eq!(f["parameters"]["b"], b);
        assert_eq!(f["parameters"]["b_post"], b_post);
        let a = KernelBinding::new()
            .predict_model(
                "gompertz",
                json!({"inputs":{"time":[10.]},"state":f["state"]}),
            )
            .unwrap()
            .result
            .unwrap();
        let expected = 200. * ((boundary / 200.).ln() * (-1.6_f64).exp()).exp();
        assert!((a["values"][0].as_f64().unwrap() - expected).abs() < 0.001);
    }
}

#[test]
fn forcing_numbers_survive_exact_json_round_trip() {
    let mut controls: Vec<f64> = (0..49).map(|i| 1. + (0.35 * i as f64).sin()).collect();
    controls.extend([f64::MIN_POSITIVE, 1e-308, f64::MAX, -0.0]);
    let encoded = serde_json::to_string(&controls).unwrap();
    let decoded: Vec<f64> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        controls.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        decoded.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn signed_zero_control_and_query_times_are_the_same_boundary() {
    let a=KernelBinding::new().predict_model("bass",json!({"inputs":{"time":[0.0]},"constructor_kwargs":{"covariates":["dose"],"projection":"cumulative_max","covariate_history":{"time":[-0.0],"values":{"dose":[1.]}}},"parameters":{"p":0.1,"q":0.,"m":100.,"beta_p_dose":0.01,"beta_q_dose":0.,"beta_m_dose":0.}})).unwrap().result.unwrap();
    assert_eq!(a["values"], json!([0.]));
    assert!((a["rates"][0][0].as_f64().unwrap() - 11.).abs() < 1e-10);
}
