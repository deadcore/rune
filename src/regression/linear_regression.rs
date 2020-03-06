use log::*;
use ndarray::{Array1, ArrayView1, ArrayView2, azip, Zip};

#[derive(Debug)]
pub struct LinearRegressionRegressor {}

impl LinearRegressionRegressor {
    pub fn new() -> Self {
        LinearRegressionRegressor {}
    }

    pub fn fit(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> LinearRegressionModel {
        let mean_y = y.mean().unwrap();
        let mean_x = x.mean().unwrap();

        let (numer, denom) = Zip::from(&y)
            .and(&x)
            .fold((0., 0.), |(numer, denom), &x, &y| {
                (numer + ((x - mean_x) * (y - mean_y)), denom + (x - mean_x).powf(2.))
            });

        let m = numer / denom;
        let c = mean_y - (m * mean_x);

        LinearRegressionModel::new(m, c)
    }
}

#[derive(Debug)]
pub struct LinearRegressionModel {
    m: f64,
    c: f64,
}

impl LinearRegressionModel {
    pub fn new(m: f64, c: f64) -> Self {
        LinearRegressionModel { m, c }
    }
    pub fn predict(&self, x: ArrayView1<f64>) -> Array1<f64> {
        x.mapv(|x| self.m * x + self.c)
    }
}