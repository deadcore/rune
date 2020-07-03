use ndarray::{Axis, ArrayView2};

use log::info;
use ndarray::prelude::*;
use rune_pipeline::pipeline::{Transformer, Fit};

#[derive(Debug)]
pub struct StandardScaler {}

pub struct StandardScalerTransformer {
    means: Array1<f64>,
    std_dev: Array1<f64>,
}

impl Transformer<ArrayView2<'_, f64>, Array2<f64>> for StandardScalerTransformer {
    fn transform(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        self.internal_transform(x)
    }
}

impl StandardScalerTransformer {
    pub fn new(means: Array1<f64>, std_dev: Array1<f64>) -> Self {
        StandardScalerTransformer {
            means,
            std_dev,
        }
    }

    pub fn internal_transform(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let xo = x.to_owned();

        (&xo - &self.means) / &self.std_dev
    }
}


impl Fit<ArrayView2<'_, f64>, StandardScalerTransformer> for StandardScaler {
    fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<bool>) -> StandardScalerTransformer {
        self.internal_fit(x)
    }
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler {}
    }

    pub fn internal_fit(&self, x: ArrayView2<f64>) -> StandardScalerTransformer {
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