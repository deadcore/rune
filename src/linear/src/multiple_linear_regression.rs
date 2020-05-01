use ndarray::{ArrayView1, ArrayView2, Array1, Axis, stack, Array2};
use log::*;

#[derive(Debug)]
pub struct MultipleLinearRegression {
    alpha: f64,
    iterations: usize,
}

#[derive(Debug)]
pub struct MultipleLinearRegressionModel {
    beta: Array1<f64>
}

impl MultipleLinearRegressionModel {
    pub fn new(beta: Array1<f64>) -> Self {
        MultipleLinearRegressionModel { beta }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let m = x.nrows();
        let x0: Array2<f64> = Array2::ones((m, 1));

        let x_with_static_coefficient = stack(Axis(1), &[x0.view(), x.view()]).unwrap();

        x_with_static_coefficient.dot(&self.beta)
    }
}

impl MultipleLinearRegression {
    pub fn new(alpha: f64, iterations: usize) -> Self {
        MultipleLinearRegression {
            alpha,
            iterations,
        }
    }

    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> MultipleLinearRegressionModel {
        let number_of_rows = x.nrows();
        let x0: Array2<f64> = Array2::ones((number_of_rows, 1));

        let x_with_static_coefficient = stack(Axis(1), &[x0.view(), x.view()]).unwrap();

        // # Initial Coefficients
        let beta: Array1<f64> = Array1::zeros(x.ncols() + 1);

        let initial_cost = self.cost(x_with_static_coefficient.view(), y, beta.view());
        debug!("initial_cost: {:#?}", initial_cost);

        let beta = self.gradient_descent(x_with_static_coefficient.view(), y, beta.view());

        MultipleLinearRegressionModel::new(beta)
    }

    fn gradient_descent(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, beta: ArrayView1<f64>) -> Array1<f64> {
        let m = y.len();

        let mut beta = beta.to_owned();

        for iteration in 0..self.iterations {
            let h = x.dot(&beta);
            trace!("[{:?}] - h: {:#?}", iteration, h);

            let loss = h - y;
            trace!("[{:?}] - loss: {:#?}", iteration, loss);

            let gradient = x.t().dot(&loss) / (m as f64);
            trace!("[{:?}] - gradient: {:#?}", iteration, gradient);

            beta = beta.to_owned() - self.alpha * gradient;
            trace!("[{:?}] - beta: {:#?}", iteration, beta);

            let cost = self.cost(x, y, beta.view());
            debug!("[{:?}] - cost: {:#?}", iteration, cost)
        }

        return beta;
    }

    pub fn cost(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, beta: ArrayView1<f64>) -> f64 {
        let m = y.len();
        (x.dot(&beta) - y).mapv(|a| a.powi(2)).sum() / (2 * m) as f64
    }
}
