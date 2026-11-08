use innovate_core::{execute_json, json};

#[test]
fn paired_outcomes_preserve_values_and_reject_misalignment() {
    let strategy = |id, value| {
        let draws: Vec<_> = (1..=3).map(|draw| json!({
            "draw_id":draw,"time":[0,1,2,3],"adoption":[0,1,2,value],
            "parameters":[{"name":"m","value":100,"unit":"people"},
                {"name":"p","value":0.03,"unit":"year^-1"},{"name":"q","value":0.3,"unit":"year^-1"}]
        })).collect();
        json!({"scenario_id":id,"time_unit":"year","perspective":"payer","provenance":"synthetic","draws":draws})
    };
    let payload = json!({"strategies":[strategy("baseline", 3), strategy("alternative", 4)]});
    let run = |payload: &serde_json::Value| {
        let request = json!({"schema_version":"2.0","operation":"validate_paired_outcomes","payload":payload});
        serde_json::from_str::<serde_json::Value>(&execute_json(&request.to_string())).unwrap()
    };
    assert_eq!(run(&payload)["result"], payload);
    let mut reordered = payload.clone();
    reordered["strategies"][1]["draws"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    assert_eq!(run(&reordered)["error"]["code"], "invalid_request");
    let mut missing = payload.clone();
    missing["strategies"][1]["draws"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert_eq!(run(&missing)["error"]["code"], "invalid_request");
    for key in ["time_unit", "perspective"] {
        let mut incompatible = payload.clone();
        incompatible["strategies"][1][key] = json!("different");
        assert_eq!(run(&incompatible)["error"]["code"], "invalid_request");
    }
    let mut negative = payload.clone();
    negative["strategies"][1]["draws"][0]["adoption"][1] = json!(-0.01);
    assert_eq!(
        run(&negative)["error"]["message"],
        "paired Bass adoption values must be nonnegative"
    );
    let mut duplicate = payload.clone();
    duplicate["strategies"][0]["draws"][0]["parameters"][1]["name"] = json!("m");
    assert_eq!(run(&duplicate)["error"]["code"], "invalid_request");
}
