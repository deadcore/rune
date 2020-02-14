use ndarray::ArrayView1;

pub mod entropy;

pub trait SelectionMeasure {
    fn apply(&self, dataset: ArrayView1<f64>, left_indexes: &[usize], right_indexes: &[usize]) -> f64;
}