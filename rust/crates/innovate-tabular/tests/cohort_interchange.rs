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
