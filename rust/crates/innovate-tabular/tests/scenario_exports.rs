use innovate_tabular::{scenario_comparisons_ipc, scenario_executions_ipc};
use polars::prelude::*;
use serde_json::{Value, json};
use std::io::Cursor;

type Exporter = fn(&Value) -> PolarsResult<Vec<u8>>;

const EXPORTERS: [(&str, Exporter); 2] = [
    ("executions", scenario_executions_ipc),
    ("comparisons", scenario_comparisons_ipc),
];

fn envelope(field: &str, rows: Value) -> Value {
    let mut response = json!({
        "operation": "execute_scenario_grid",
        "result": {},
        "error": null,
    });
    response["result"][field] = rows;
    response
}

fn execution() -> Value {
    json!({"id": "baseline", "model_key": "bass", "result": {}, "error": null})
}

fn pair() -> Value {
    json!({"baseline": "a", "alternative": "b", "result": {}, "error": null})
}

fn ensemble() -> Value {
    json!({"pairs": [{"baseline": "a", "alternative": "b"}], "result": {}, "error": null})
}

fn read_export(export: Exporter, response: &Value) -> DataFrame {
    IpcReader::new(Cursor::new(export(response).unwrap()))
        .finish()
        .unwrap()
}

fn assert_json_rows(frame: &DataFrame, column: &str, rows: &Value) {
    let encoded = frame.column(column).unwrap().str().unwrap();
    assert_eq!(frame.height(), rows.as_array().unwrap().len());
    for (index, expected) in rows.as_array().unwrap().iter().enumerate() {
        let actual: Value = serde_json::from_str(encoded.get(index).unwrap()).unwrap();
        assert_eq!(&actual, expected);
    }
}

#[test]
fn scenario_exports_require_successful_explicit_envelope_outcomes() {
    for (field, export) in EXPORTERS {
        let row = if field == "executions" {
            execution()
        } else {
            pair()
        };
        let valid = envelope(field, json!([row]));
        for missing in ["error", "result"] {
            let mut response = valid.clone();
            response.as_object_mut().unwrap().remove(missing);
            assert!(
                export(&response).is_err(),
                "{field}: accepted missing {missing}"
            );
        }
        for invalid in [
            Value::Null,
            json!(false),
            json!(7),
            json!("result"),
            json!([]),
        ] {
            let mut response = valid.clone();
            response["result"] = invalid.clone();
            assert!(
                export(&response).is_err(),
                "{field}: accepted result {invalid}"
            );
            assert!(
                export(&invalid).is_err(),
                "{field}: accepted envelope {invalid}"
            );
        }
        for invalid in [json!({}), json!(false), json!(7), json!("error"), json!([])] {
            let mut response = valid.clone();
            response["error"] = invalid.clone();
            assert!(
                export(&response).is_err(),
                "{field}: accepted envelope error {invalid}"
            );
        }
        let failed = json!({
            "operation": "execute_scenario_grid",
            "result": null,
            "error": {"code": "invalid_request"},
        });
        assert!(export(&failed).is_err());
    }
}

fn assert_missing_outcomes_rejected(field: &str, export: Exporter, row: Value) {
    for missing in [vec!["result"], vec!["error"], vec!["result", "error"]] {
        let mut invalid = row.clone();
        for key in &missing {
            invalid.as_object_mut().unwrap().remove(*key);
        }
        let response = envelope(field, json!([row.clone(), invalid]));
        assert!(
            export(&response).is_err(),
            "{field}: accepted row missing {missing:?}"
        );
    }
}

#[test]
fn scenario_execution_exports_require_explicit_row_outcomes() {
    assert_missing_outcomes_rejected("executions", scenario_executions_ipc, execution());
}

#[test]
fn scenario_comparison_exports_require_explicit_pair_and_ensemble_outcomes() {
    for row in [pair(), ensemble()] {
        assert_missing_outcomes_rejected("comparisons", scenario_comparisons_ipc, row);
    }
}

fn assert_invalid_outcomes_rejected(field: &str, export: Exporter, row: Value) {
    let cases = [
        (json!(null), json!(null)),
        (json!({}), json!({"code": "invalid_request"})),
        (json!(false), json!(null)),
        (json!(7), json!(null)),
        (json!("result"), json!(null)),
        (json!([]), json!(null)),
        (json!(null), json!(false)),
        (json!(null), json!(7)),
        (json!(null), json!("error")),
        (json!(null), json!([])),
    ];
    for (result, error) in cases {
        let mut invalid = row.clone();
        invalid["result"] = result;
        invalid["error"] = error;
        let response = envelope(field, json!([row.clone(), invalid.clone()]));
        assert!(
            export(&response).is_err(),
            "{field}: accepted invalid outcome {invalid}"
        );
    }
    for invalid in [json!(null), json!(false), json!(7), json!("row"), json!([])] {
        let response = envelope(field, json!([row.clone(), invalid.clone()]));
        assert!(
            export(&response).is_err(),
            "{field}: accepted non-object row {invalid}"
        );
    }
}

#[test]
fn scenario_execution_exports_reject_contradictory_and_mistyped_outcomes() {
    assert_invalid_outcomes_rejected("executions", scenario_executions_ipc, execution());
}

#[test]
fn scenario_comparison_exports_reject_contradictory_and_mistyped_outcomes() {
    for row in [pair(), ensemble()] {
        assert_invalid_outcomes_rejected("comparisons", scenario_comparisons_ipc, row);
    }
}

#[test]
fn scenario_execution_exports_preserve_success_failure_pause_and_nested_json() {
    let rows = json!([
        {
            "id": "baseline-β", "model_key": "bass",
            "result": {"counts": [0, 2], "metadata": {"seed": u64::MAX, "note": null}},
            "error": null, "artifact": {"provenance": "synthetic α"},
        },
        {
            "id": "failed", "model_key": "unknown", "result": null,
            "error": {"code": "unsupported_native_operation", "message": "unavailable β"},
        },
        {
            "id": "paused-γ", "model_key": "bass",
            "result": {"status": "paused", "checkpoint": {"seed": u64::MAX}},
            "error": null,
        },
    ]);
    let response = envelope("executions", rows.clone());
    let frame = read_export(scenario_executions_ipc, &response);
    assert_eq!(
        frame
            .column("scenario_id")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some("baseline-β"), Some("failed"), Some("paused-γ")]
    );
    assert_eq!(
        frame
            .column("successful")
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some(true), Some(false), Some(false)]
    );
    assert_json_rows(&frame, "execution_json", &rows);
}

#[test]
fn scenario_comparison_exports_preserve_pair_ensemble_outcomes_and_nested_json() {
    let rows = json!([
        {
            "baseline": "a", "alternative": "β",
            "result": {"difference": [2.5], "metadata": {"seed": u64::MAX, "note": null}},
            "error": null,
        },
        {
            "baseline": "a", "alternative": "missing", "result": null,
            "error": {"code": "invalid_comparison", "message": "missing β"},
        },
        {
            "pairs": [{"baseline": "a", "alternative": "β"}],
            "result": {"mean_difference": [2.5]}, "error": null,
        },
        {
            "pairs": [{"baseline": "a", "alternative": "missing"}],
            "result": null, "error": {"code": "invalid_comparison"},
        },
    ]);
    let response = envelope("comparisons", rows.clone());
    let frame = read_export(scenario_comparisons_ipc, &response);
    assert_eq!(
        frame
            .column("comparison_index")
            .unwrap()
            .u64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(
        frame
            .column("kind")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [
            Some("pair"),
            Some("pair"),
            Some("ensemble"),
            Some("ensemble")
        ]
    );
    assert_eq!(
        frame
            .column("successful")
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [Some(true), Some(false), Some(true), Some(false)]
    );
    assert_json_rows(&frame, "comparison_json", &rows);
}

#[test]
fn scenario_exports_recover_after_rejected_input() {
    for (field, export) in EXPORTERS {
        let row = if field == "executions" {
            execution()
        } else {
            pair()
        };
        let valid = envelope(field, json!([row]));
        let before = read_export(export, &valid);
        let mut invalid = valid.clone();
        invalid["result"][field][0]["result"] = Value::Null;
        assert!(
            export(&invalid).is_err(),
            "{field}: accepted an empty outcome"
        );
        assert_eq!(read_export(export, &valid), before);
    }
}
