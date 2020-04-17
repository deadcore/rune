use ndarray::{ArrayView1, Zip};

pub fn r2(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f64 {
    let mean_y = y_true.mean().unwrap();

    let (ss_t, ss_r) = Zip::from(&y_true)
        .and(&y_pred)
        .fold((0., 0.), |(ss_t, ss_r), y_true, y_pred| {
            (ss_t + (y_true - mean_y).powf(2.), ss_r + (y_true - y_pred).powf(2.))
        });
    1. - (ss_r / ss_t)
}