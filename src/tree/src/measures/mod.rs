use ndarray::ArrayView1;
use std::hash::Hash;

pub mod entropy;

pub trait SelectionMeasure {
    fn apply<T: Copy + Eq + Hash>(&self, dataset: ArrayView1<T>, left_indexes: &[usize], right_indexes: &[usize]) -> f64;
}