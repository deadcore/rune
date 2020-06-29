use ndarray::{ArrayView1, Axis, ArrayView2};

use log::info;
use ndarray_stats::CorrelationExt;
use ndarray_stats::errors::EmptyInput;
use std::error::Error;
use ndarray::prelude::*;

#[derive(Debug)]
pub struct StandardScaler {}

pub struct StandardScalerTransformer {
    means: Array1<f64>,
    std_dev: Array1<f64>,
}

impl StandardScalerTransformer {
    pub fn new(means: Array1<f64>, std_dev: Array1<f64>) -> Self {
        StandardScalerTransformer {
            means,
            std_dev,
        }
    }

    pub fn transform(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let xo = x.to_owned();

        (&xo - &self.means) / &self.std_dev
    }
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler {}
    }

    pub fn fit(&self, x: ArrayView2<f64>) -> StandardScalerTransformer {
        let xo = x.to_owned();
        let mean: &Array1<f64> = &xo.mean_axis(Axis(0)).unwrap();
        let std_dev: &Array1<f64> = &xo.std_axis(Axis(0), 1.);
        let std_scale = (&xo - mean) / std_dev;

        info!("mean: {}", mean);
        info!("std_dev: {}", std_dev);
        info!("std_scale: {}", std_scale);

        StandardScalerTransformer::new(
            mean.to_owned(),
            std_dev.to_owned(),
        )
    }
}