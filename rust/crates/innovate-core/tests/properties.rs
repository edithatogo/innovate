use innovate_core::domain::bass_cumulative;
use innovate_core::models::norton_bass::norton_bass_two_gen;

#[test]
fn bass_cumulative_is_monotone_and_bounded_across_parameter_grid() {
    for &p in &[1e-4, 0.01, 0.05] {
        for &q in &[0.0, 0.1, 0.5] {
            for &m in &[1.0, 500.0, 1e5] {
                let mut prev = 0.0;
                for step in 0..=40 {
                    let t = step as f64 * 0.5;
                    let val = bass_cumulative(t, p, q, m).unwrap();
                    assert!(val >= prev && val <= m && val.is_finite());
                    prev = val;
                }
            }
        }
    }
}

#[test]
fn norton_bass_conservation_holds_across_launch_times() {
    for &tau2 in &[0.0, 2.5, 10.0] {
        for step in 0..=20 {
            let t = step as f64;
            let [s1, s2] =
                norton_bass_two_gen(t, 0.03, 0.38, 1000.0, 0.04, 0.4, 500.0, tau2).unwrap();
            assert!(s1 >= 0.0 && s2 >= 0.0);
            let f1 = bass_cumulative(t, 0.03, 0.38, 1000.0).unwrap();
            let f2 = if t >= tau2 {
                bass_cumulative(t - tau2, 0.04, 0.4, 500.0).unwrap()
            } else {
                0.0
            };
            assert!((s1 + s2 - (f1 + f2)).abs() < 1e-9);
        }
    }
}
