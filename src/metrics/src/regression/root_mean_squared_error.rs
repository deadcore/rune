use ndarray::{ArrayView1, Zip};

pub fn root_mean_squared_error(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f64 {
    let rmse = Zip::from(&y_true)
        .and(&y_pred)
        .fold(0., |acc, y_true, y_pred| {
            acc + (y_true - y_pred).powf(2.)
        });

    (rmse / y_true.len() as f64).sqrt()
}