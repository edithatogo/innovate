use innovate_core::{KernelBinding, json};
fn bass(t: f64, p: f64, q: f64, m: f64) -> f64 {
    let decay = (-(p + q) * t).exp();
    m * (1. - decay) / (1. + q / p * decay)
}
#[test]
fn historical_weighted_spillover_preserves_isolated_nodes_and_matrix_shape() {
    let time = [0_f64, 1., 5., 20.];
    let r=KernelBinding::new().predict_model("network_diffusion",json!({"inputs":{"time":time},"constructor_kwargs":{"base_model":"bass","adjacency":[[0.,2.,1.],[1.,0.,0.],[0.,0.,0.]],"node_labels":["A","B","isolated"]},"parameters":{"spillover_strength":0.2,"node_0_p":0.03,"node_0_q":0.3,"node_0_m":100.,"node_1_p":0.02,"node_1_q":0.2,"node_1_m":200.,"node_2_p":0.04,"node_2_q":0.4,"node_2_m":50.}})).unwrap().result.unwrap();
    assert_eq!(r["columns"], json!(["A", "B", "isolated"]));
    for (i, t) in time.iter().enumerate() {
        let a = bass(*t, 0.03, 0.3, 100.);
        let b = bass(*t, 0.02, 0.2, 200.);
        let c = bass(*t, 0.04, 0.4, 50.);
        for (j, v) in [a + 0.2 * (2. * b + c) / 3., b + 0.2 * a, c]
            .iter()
            .enumerate()
        {
            assert!((r["rows"][i][j].as_f64().unwrap() - v).abs() < 1e-8);
        }
    }
}
#[test]
fn independent_node_fits_retain_state_and_forecast() {
    let t: Vec<_> = (0..26).map(|i| i as f64).collect();
    let observed: Vec<Vec<_>> = vec![
        t.iter().map(|t| bass(*t, 0.03, 0.3, 100.)).collect(),
        t.iter().map(|t| bass(*t, 0.02, 0.2, 200.)).collect(),
    ];
    let fit=KernelBinding::new().fit_model("network_diffusion",json!({"inputs":{"time":t,"observed":observed},"constructor_kwargs":{"adjacency":[[0.,1.],[1.,0.]],"orientation":"node_time","spillover_strength":0.}})).unwrap().result.unwrap();
    assert_eq!(fit["node_diagnostics"].as_array().unwrap().len(), 2);
    assert_eq!(
        fit["parameters"]["spillover_strength"].as_f64().unwrap(),
        0.
    );
    let a = KernelBinding::new()
        .predict_model(
            "network_diffusion",
            json!({"inputs":{"time":[30.]},"state":fit["state"]}),
        )
        .unwrap()
        .result
        .unwrap();
    assert!((a["rows"][0][0].as_f64().unwrap() - bass(30., 0.03, 0.3, 100.)).abs() < 0.01);
    assert!((a["rows"][0][1].as_f64().unwrap() - bass(30., 0.02, 0.2, 200.)).abs() < 0.01);
}
#[test]
fn graph_shape_and_unknown_options_are_rejected() {
    for kw in [
        json!({"adjacency":[[0.,1.]]}),
        json!({"adjacency":[[0.]],"node_labels":["a","b"]}),
        json!({"adjacency":[[0.]],"unknown":true}),
    ] {
        assert!(KernelBinding::new().predict_model("network_diffusion",json!({"inputs":{"time":[0.,1.]},"constructor_kwargs":kw,"parameters":{"spillover_strength":0.,"node_0_p":0.03,"node_0_q":0.3,"node_0_m":100.}})).is_err());
    }
}
#[test]
fn historical_summary_retains_coordinates_and_graph_statistics_without_times() {
    let r=KernelBinding::new().summarize_model("network_diffusion",json!({"constructor_kwargs":{"adjacency":[[9.,2.],[1.,9.]],"coordinates":[[0.,0.],[1.,1.]]},"parameters":{"spillover_strength":0.2,"node_0_p":0.03,"node_0_q":0.3,"node_0_m":100.,"node_1_p":0.02,"node_1_q":0.2,"node_1_m":200.}})).unwrap().result.unwrap();
    assert_eq!(r["details"]["edge_count"], 2);
    assert_eq!(r["details"]["edge_density"], 1.);
    assert_eq!(r["details"]["average_degree"], 1.5);
    assert_eq!(
        r["state"]["constructor_kwargs"]["adjacency"],
        json!([[0., 2.], [1., 0.]])
    );
    assert_eq!(r["details"]["coordinates"], json!([[0., 0.], [1., 1.]]));
    assert!(r.get("predictions").is_none());
}
#[test]
fn finite_high_weight_graphs_keep_finite_summary_means() {
    let r=KernelBinding::new().summarize_model("network_diffusion",json!({"constructor_kwargs":{"adjacency":[[0.,1e308],[1e308,0.]]},"parameters":{"spillover_strength":0.2,"node_0_p":0.03,"node_0_q":0.3,"node_0_m":100.,"node_1_p":0.02,"node_1_q":0.2,"node_1_m":200.}})).unwrap().result.unwrap();
    assert_eq!(r["details"]["average_degree"].as_f64(), Some(1e308));
}
