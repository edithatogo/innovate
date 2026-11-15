//! Preserve bootstrap missingness and scale behavior during the statistics migration.
use super::*;

#[test]
fn statrs_migration_correlation_preserves_extreme_scale_and_sign() {
    for scale in [1e-300, 1., 1e300] {
        let draws = BTreeMap::from([
            ("increasing".into(), vec![scale, 2. * scale, 3. * scale]),
            ("decreasing".into(), vec![3. * scale, 2. * scale, scale]),
            ("zero".into(), vec![0.; 3]),
        ]);
        let matrix = correlation(&draws, 3);
        assert!((matrix["increasing"]["decreasing"].as_f64().unwrap() + 1.).abs() < 1e-14);
        assert!((matrix["increasing"]["increasing"].as_f64().unwrap() - 1.).abs() < 1e-14);
        assert_eq!(
            matrix["increasing"]["decreasing"],
            matrix["decreasing"]["increasing"]
        );
        assert!(matrix["increasing"]["zero"].is_null());
        assert!(matrix["zero"]["zero"].is_null());
    }
}

#[test]
fn statrs_migration_correlation_does_not_invent_insufficient_draws() {
    let one_parameter = BTreeMap::from([("a".into(), vec![1., 2.])]);
    assert!(correlation(&one_parameter, 2).is_null());
    let one_draw = BTreeMap::from([("a".into(), vec![1.]), ("b".into(), vec![2.])]);
    assert!(correlation(&one_draw, 1).is_null());
    assert!(correlation(&BTreeMap::new(), 0).is_null());
}

#[test]
fn statrs_migration_constant_columns_remain_null_not_zero() {
    let draws = BTreeMap::from([
        ("positive".into(), vec![7.; 4]),
        ("negative".into(), vec![-9.; 4]),
    ]);
    let matrix = correlation(&draws, 4);
    assert_eq!(
        matrix,
        json!({
            "positive":{"positive":null,"negative":null},
            "negative":{"positive":null,"negative":null}
        })
    );
}
