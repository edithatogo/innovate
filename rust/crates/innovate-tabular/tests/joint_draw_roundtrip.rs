use innovate_tabular::{JointDrawBundle, JointDrawRow, decode_joint_draws, encode_joint_draws};

fn sample_bundle() -> JointDrawBundle {
    JointDrawBundle {
        schema_version: "1.0".into(),
        dependence: "paired_joint_posterior".into(),
        rows: vec![
            JointDrawRow {
                draw_id: 0,
                chain_id: 0,
                scenario_id: "base".into(),
                strategy_id: "status_quo".into(),
                product: "therapy_a".into(),
                region: "AU".into(),
                time: 0.0,
                value: None,
                units: "count".into(),
                fit_status: "valid".into(),
                provenance: "innovate-core:nuts".into(),
            },
            JointDrawRow {
                draw_id: 0,
                chain_id: 0,
                scenario_id: "base".into(),
                strategy_id: "status_quo".into(),
                product: "therapy_a".into(),
                region: "AU".into(),
                time: 1.0,
                value: Some(42.5),
                units: "count".into(),
                fit_status: "valid".into(),
                provenance: "innovate-core:nuts".into(),
            },
        ],
    }
}

#[test]
fn joint_draw_ipc_roundtrips_with_nulls_and_metadata() {
    let bundle = sample_bundle();
    let bytes = encode_joint_draws(&bundle).expect("encode valid bundle");
    let decoded = decode_joint_draws(&bytes).expect("decode valid bundle");
    assert_eq!(decoded, bundle);
}

#[test]
fn joint_draw_rejects_unsupported_schema_duplicate_keys_missing_units_and_invalid_fit() {
    let mut bad_schema = sample_bundle();
    bad_schema.schema_version = "9.9".into();
    assert!(encode_joint_draws(&bad_schema).is_err());

    let mut dup_key = sample_bundle();
    dup_key.rows[1].time = 0.0;
    assert!(encode_joint_draws(&dup_key).is_err());

    let mut missing_units = sample_bundle();
    missing_units.rows[0].units = " ".into();
    assert!(encode_joint_draws(&missing_units).is_err());

    let mut invalid_fit = sample_bundle();
    invalid_fit.rows[1].fit_status = "failed".into();
    assert!(encode_joint_draws(&invalid_fit).is_err());
}
