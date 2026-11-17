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
