use ndarray::{ArrayView1, Axis, ArrayView2};

use log::info;
use ndarray_stats::CorrelationExt;
use ndarray_stats::errors::EmptyInput;
use ndarray_linalg::{Eigh, UPLO};
use std::error::Error;

#[derive(Debug)]
pub struct PrincipalComponentAnalysis {
    k: usize
}

pub struct PrincipalComponentAnalysisTransformer {}

impl PrincipalComponentAnalysisTransformer {
    pub fn new() -> Self {
        PrincipalComponentAnalysisTransformer {}
    }
}

impl PrincipalComponentAnalysis {
    pub fn new(k: usize) -> Self {
        PrincipalComponentAnalysis {
            k
        }
    }

    pub fn fit(&self, x: ArrayView2<f64>) -> Result<PrincipalComponentAnalysisTransformer, Box<dyn Error>> {
        let co_variance_matrix = x.t().cov(1.)?;
        info!("co_variance_matrix: \n {}", co_variance_matrix);

        let (eig_val, eig_vec) = co_variance_matrix.eigh(UPLO::Upper)?;
        info!("eig_val: {}", eig_val);
        info!("eig_vec: {}", eig_vec);



        Ok(PrincipalComponentAnalysisTransformer::new())
    }
}