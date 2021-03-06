use ndarray::{Axis, ArrayView2, Array2, stack, Array, ArrayView1};

use log::debug;
use ndarray_stats::CorrelationExt;
use ndarray_linalg::{Eigh, UPLO};
use std::error::Error;
use std::cmp::Ordering;
use rune_pipeline::pipeline::{Transformer, Fit};

#[derive(Debug)]
pub struct PrincipalComponentAnalysis {
    number_of_features: usize
}

pub struct PrincipalComponentAnalysisTransformer {
    projection: Array2<f64>,
}

impl Transformer<ArrayView2<'_, f64>, Array2<f64>> for PrincipalComponentAnalysisTransformer {
    fn transform(&self, x: ArrayView2<'_, f64>) -> Array2<f64> {
        self.internal_transform(x)
    }
}

impl Transformer<Array2<f64>, Array2<f64>> for PrincipalComponentAnalysisTransformer {
    fn transform(&self, x: Array2<f64>) -> Array2<f64> {
        self.internal_transform(x.view())
    }
}


impl Fit<ArrayView2<'_, f64>, PrincipalComponentAnalysisTransformer> for PrincipalComponentAnalysis {
    fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<bool>) -> PrincipalComponentAnalysisTransformer {
        self.internal_fit(x).unwrap()
    }
}

impl Fit<Array2<f64>, PrincipalComponentAnalysisTransformer> for PrincipalComponentAnalysis {
    fn fit(&self, x: Array2<f64>, y: ArrayView1<bool>) -> PrincipalComponentAnalysisTransformer {
        self.internal_fit(x.view()).unwrap()
    }
}

impl PrincipalComponentAnalysisTransformer {
    pub fn new(projection: Array2<f64>) -> Self {
        PrincipalComponentAnalysisTransformer { projection }
    }

    pub fn internal_transform(&self, x: ArrayView2<f64>) -> Array2<f64> {
        return x.dot(&self.projection);
    }
}

impl PrincipalComponentAnalysis {
    pub fn new(number_of_features: usize) -> Self {
        PrincipalComponentAnalysis {
            number_of_features
        }
    }

    pub fn internal_fit(&self, x: ArrayView2<f64>) -> Result<PrincipalComponentAnalysisTransformer, Box<dyn Error>> {
        let co_variance_matrix = x.t().cov(1.)?;
        debug!("co_variance_matrix: \n {}", co_variance_matrix);

        // eig_vec: The vector which is only stretched or squashed
        // eig_val: The amount that vector is stretched or squashed
        let (eig_val, eig_vec) = co_variance_matrix.eigh(UPLO::Upper)?;
        debug!("eig_val: {}", eig_val);
        debug!("eig_vec: {}", eig_vec);

        let mut arr = Vec::new();

        for i in 0..eig_val.len() {
            arr.push((eig_val[i], eig_vec.column(i)))
        }

        arr.sort_by(|o1, o2|
            if o1.0 < o2.0 {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );

        for i in 0..arr.len() {
            debug!("arr[{}]: {:?}", i, arr[i]);
        }

        let mut z: Vec<Array2<f64>> = Vec::new();

        for i in &arr[0..2] {
            let v = i.1;
            let x = Array::from_shape_vec((4, 1), v.to_vec())?;
            z.push(x);
        }

        debug!("arr[]: {:?}", z);

        let x = z.iter().map(|x| x.view()).collect::<Vec<ArrayView2<f64>>>();


        let projection: Array2<f64> = stack(Axis(1), &x[0..self.number_of_features])?;

        debug!("feature_projection: {:?}", projection);


        Ok(PrincipalComponentAnalysisTransformer::new(projection))
    }
}