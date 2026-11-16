use innovate_tabular::{ColumnBinding, bind_panel};
use polars_core::prelude::*;

#[test]
fn bind_panel_preserves_nulls_and_is_invariant_to_column_permutation() {
    let binding = ColumnBinding {
        time: "t".into(),
        product: "prod".into(),
        measure: "meas".into(),
        unit: "u".into(),
        value: "val".into(),
        denominator: Some("pop".into()),
    };

    let df_a = df!(
        "t" => [0.0, 1.0, 2.0],
        "prod" => ["ev", "ev", "ev"],
        "meas" => ["share", "share", "share"],
        "u" => ["proportion", "proportion", "proportion"],
        "val" => [None, Some(0.0), Some(0.25)],
        "pop" => [Some(1000.0), Some(1000.0), Some(1000.0)]
    )
    .unwrap();

    let df_b = df!(
        "pop" => [Some(1000.0), Some(1000.0), Some(1000.0)],
        "val" => [None, Some(0.0), Some(0.25)],
        "u" => ["proportion", "proportion", "proportion"],
        "meas" => ["share", "share", "share"],
        "prod" => ["ev", "ev", "ev"],
        "t" => [0.0, 1.0, 2.0]
    )
    .unwrap();

    let panel_a = bind_panel(df_a, &binding).unwrap();
    let panel_b = bind_panel(df_b, &binding).unwrap();
    assert_eq!(panel_a, panel_b);
    assert_eq!(panel_a.rows[0].value, None);
    assert_eq!(panel_a.rows[1].value, Some(0.0));
}

#[test]
fn bind_panel_rejects_missing_columns_duplicate_keys_and_ambiguous_units() {
    let binding = ColumnBinding {
        time: "t".into(),
        product: "prod".into(),
        measure: "meas".into(),
        unit: "u".into(),
        value: "val".into(),
        denominator: None,
    };

    let missing_col = df!("t" => [0.0], "prod" => ["a"]).unwrap();
    assert!(bind_panel(missing_col, &binding).is_err());

    let duplicate_key = df!(
        "t" => [1.0, 1.0],
        "prod" => ["a", "a"],
        "meas" => ["m", "m"],
        "u" => ["count", "count"],
        "val" => [Some(1.0), Some(2.0)]
    )
    .unwrap();
    assert!(bind_panel(duplicate_key, &binding).is_err());

    let ambiguous_unit = df!(
        "t" => [1.0, 2.0],
        "prod" => ["a", "a"],
        "meas" => ["m", "m"],
        "u" => ["count", "proportion"],
        "val" => [Some(1.0), Some(0.2)]
    )
    .unwrap();
    assert!(bind_panel(ambiguous_unit, &binding).is_err());
}
