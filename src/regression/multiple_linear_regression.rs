use ndarray::{ArrayView1, ArrayView2, Array1};

#[derive(Debug)]
pub struct MultipleLinearRegressionClassifier {
    alpha: f64
}

impl MultipleLinearRegressionClassifier {
    pub fn new(alpha: f64) -> Self {
        MultipleLinearRegressionClassifier {
            alpha
        }
    }

    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) {
        // math = data['Math'].values
        // read = data['Reading'].values
        // write = data['Writing'].values

        m = x.nrows();
        x0 = Array1::zeros(m);

        X = np.array([x0, math, read]).T
        // # Initial Coefficients
        B = np.array([0, 0, 0]);
    }

    pub fn cost(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> f64 {0.}
}
