//! Public calendar controls for evaluating temporal dependency migrations.
//!
//! Expectations are fixed Gregorian/RFC 3339 examples rather than values
//! calculated with the dependency that the implementation currently uses.

use innovate_core::json;
use serde_json::Value;

fn aggregate(dates: Value, values: Value, frequency: &str) -> Value {
    let request = json!({
        "schema_version": "2.0",
        "operation": "aggregate_dates",
        "payload": {"dates": dates, "values": values, "frequency": frequency}
    });
    serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap()
}

#[test]
fn expanded_years_preserve_gregorian_leap_days_and_maximum_date() {
    let leap = aggregate(json!(["+10000-02-28", "+10000-03-01"]), json!([3, 4]), "D");
    assert_eq!(
        leap["result"],
        json!({
            "dates": ["+10000-02-28", "+10000-02-29", "+10000-03-01"],
            "values": [3.0, 0.0, 4.0]
        })
    );

    let last = aggregate(json!(["+262142-12-31"]), json!([1]), "ME");
    assert_eq!(
        last["result"],
        json!({"dates": ["+262142-12-31"], "values": [1.0]})
    );

    let impossible = aggregate(json!(["+10000-02-30"]), json!([1]), "D");
    assert_eq!(impossible["error"]["code"], "invalid_request");
}

#[test]
fn calendar_dates_accept_unpadded_fields_and_reject_leading_whitespace() {
    let response = aggregate(json!(["2024-1-1", "2024-01-01"]), json!([1, 2]), "D");
    assert_eq!(
        response["result"],
        json!({"dates": ["2024-01-01"], "values": [3.0]})
    );
    let invalid = aggregate(json!([" 2024-01-01"]), json!([1]), "D");
    assert_eq!(invalid["error"]["code"], "invalid_request");
}

#[test]
fn offset_timestamps_are_bucketed_by_utc_date() {
    let response = aggregate(
        json!(["2024-01-01T23:30:00-02:00", "2024-01-02T00:30:00+01:00"]),
        json!([3, 5]),
        "D",
    );
    assert_eq!(
        response["result"],
        json!({"dates": ["2024-01-01", "2024-01-02"], "values": [5.0, 3.0]})
    );
}

#[test]
fn fractional_weekly_anchor_preserves_the_exact_seven_day_boundary() {
    let response = aggregate(
        json!([
            "2024-01-01T12:00:00.123456Z",
            "2024-01-08T12:00:00.123455Z",
            "2024-01-08T12:00:00.123456Z"
        ]),
        json!([1, 2, 4]),
        "W",
    );
    assert_eq!(
        response["result"],
        json!({
            "dates": ["2024-01-01T12:00:00.123456Z", "2024-01-08T12:00:00.123456Z"],
            "values": [3.0, 4.0]
        })
    );
}

#[test]
fn leap_second_label_is_not_silently_normalized() {
    let response = aggregate(json!(["2016-12-31T23:59:60Z"]), json!([1]), "W");
    assert_eq!(
        response["result"],
        json!({"dates": ["2016-12-31T23:59:60Z"], "values": [1.0]})
    );
    let invalid = aggregate(json!(["2016-12-31T23:59:61Z"]), json!([1]), "W");
    assert_eq!(invalid["error"]["code"], "invalid_request");
}

#[test]
fn naive_timestamps_keep_naive_labels_and_cannot_mix_with_offsets() {
    let naive = aggregate(
        json!(["2024-01-01 12:00:00", "2024-01-08T12:00:00"]),
        json!([1, 2]),
        "W",
    );
    assert_eq!(
        naive["result"],
        json!({"dates": ["2024-01-01T12:00:00", "2024-01-08T12:00:00"], "values": [1.0, 2.0]})
    );

    let mixed = aggregate(
        json!(["2024-01-01T12:00:00", "2024-01-02T12:00:00Z"]),
        json!([1, 2]),
        "D",
    );
    assert_eq!(mixed["error"]["code"], "invalid_request");
    assert_eq!(
        mixed["error"]["message"],
        "cannot mix offset timestamps and naive timestamps"
    );
}

#[test]
fn imported_artifacts_preserve_rfc3339_timestamps_and_reject_invalid_forms() {
    let mut request = json!({
        "schema_version": "2.0",
        "operation": "import_scenario_artifact",
        "payload": {"artifact": {
            "scenario": {
                "scenario_type": "baseline", "name": "baseline", "description": "",
                "time_horizon": 5, "time_unit": "year", "reference_year": 2026,
                "market_size": 100, "initial_adoption": 0
            },
            "seed": 42, "model_type": "bass", "version": "0.6.0", "notes": null,
            "timestamp": "2026-01-01T00:00:00+00:00"
        }}
    });
    for timestamp in [
        "2026-01-01T00:00:00+00:00",
        "2024-02-29T15:04:05.123456+10:00",
        "2016-12-31T23:59:60Z",
    ] {
        request["payload"]["artifact"]["timestamp"] = json!(timestamp);
        let response: Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert!(response["error"].is_null(), "{timestamp}: {response}");
        assert_eq!(response["result"]["artifact"]["timestamp"], timestamp);
        assert_eq!(response["result"]["replay_available"], false);
    }
    for timestamp in [
        "2026-01-01T00:00:00",
        "2023-02-29T00:00:00Z",
        "2016-12-31T23:59:61Z",
        "2026-01-01T00:00:00+24:00",
    ] {
        request["payload"]["artifact"]["timestamp"] = json!(timestamp);
        let response: Value =
            serde_json::from_str(&innovate_core::execute_json(&request.to_string())).unwrap();
        assert_eq!(response["error"]["code"], "invalid_request", "{timestamp}");
        assert_eq!(
            response["error"]["message"],
            "scenario artifact timestamp must be RFC 3339 with a time offset",
            "{timestamp}"
        );
    }
}
