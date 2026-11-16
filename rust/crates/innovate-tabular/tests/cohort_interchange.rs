use innovate_tabular::{CohortOccupancyRow, cohort_occupancy_ipc, decode_cohort_occupancy_ipc};

#[test]
fn cohort_occupancy_ipc_roundtrips_with_all_keys() {
    let rows = vec![
        CohortOccupancyRow {
            draw_id: 0,
            scenario_id: "baseline".into(),
            cohort_id: "1990_birth".into(),
            subgroup: "all".into(),
            state: "former".into(),
            duration_bin: 0,
            time: 1.0,
            occupancy: 120.0,
            transition_flow: 15.0,
        },
        CohortOccupancyRow {
            draw_id: 0,
            scenario_id: "baseline".into(),
            cohort_id: "1990_birth".into(),
            subgroup: "all".into(),
            state: "former".into(),
            duration_bin: 1,
            time: 2.0,
            occupancy: 105.0,
            transition_flow: 8.0,
        },
    ];
    let bytes = cohort_occupancy_ipc(&rows).expect("encode cohort occupancy");
    let decoded = decode_cohort_occupancy_ipc(&bytes).expect("decode cohort occupancy");
    assert_eq!(decoded, rows);

    assert!(cohort_occupancy_ipc(&[]).is_err());
}

#[test]
fn cohort_ipc_rejects_nulls_in_each_required_column() {
    use polars_arrow::io::ipc::write::{FileWriter, WriteOptions};
    use polars_core::prelude::*;

    for missing in [
        "draw_id",
        "scenario_id",
        "cohort_id",
        "subgroup",
        "state",
        "duration_bin",
        "time",
        "occupancy",
        "transition_flow",
    ] {
        let mut frame = df!(
            "draw_id" => [Some(7_u64), (missing != "draw_id").then_some(8_u64)],
            "scenario_id" => [Some("base"), (missing != "scenario_id").then_some("base")],
            "cohort_id" => [Some("cohort"), (missing != "cohort_id").then_some("cohort")],
            "subgroup" => [Some("all"), (missing != "subgroup").then_some("all")],
            "state" => [Some("former"), (missing != "state").then_some("former")],
            "duration_bin" => [Some(0_u32), (missing != "duration_bin").then_some(1_u32)],
            "time" => [Some(0.0_f64), (missing != "time").then_some(1.0_f64)],
            "occupancy" => [Some(100.0_f64), (missing != "occupancy").then_some(90.0_f64)],
            "transition_flow" => [Some(0.0_f64), (missing != "transition_flow").then_some(10.0_f64)]
        )
        .unwrap();
        let compat = CompatLevel::oldest();
        let schema = frame.schema().to_arrow(compat);
        frame.align_chunks_par();
        let mut bytes = Vec::new();
        let mut writer = FileWriter::new(
            &mut bytes,
            schema.into(),
            None,
            WriteOptions { compression: None },
        );
        writer.start().unwrap();
        for batch in frame.iter_chunks(compat, true) {
            writer.write(&batch, None).unwrap();
        }
        writer.finish().unwrap();

        let err = decode_cohort_occupancy_ipc(&bytes).unwrap_err();
        assert!(
            matches!(err, PolarsError::ComputeError(_)),
            "{missing}: {err}"
        );
    }
}
