use std::collections::HashMap;

use ndarray::{ArrayView1, Axis};

use crate::measures::SelectionMeasure;

#[derive(Debug)]
pub struct EntropySelectionMeasure {}

impl SelectionMeasure for EntropySelectionMeasure {
    fn apply(&self, dataset: ArrayView1<f64>, left_indexes: &[usize], right_indexes: &[usize]) -> f64 {
        let total_entropy = entropy(dataset);
        let left_entropy = entropy(dataset.select(Axis(0), left_indexes).view());
        let right_entropy = entropy(dataset.select(Axis(0), right_indexes).view());

        let weighted_left_entropy = (left_indexes.len() as f64 / dataset.len() as f64) as f64 * left_entropy;
        let weighted_right_entropy = (right_indexes.len() as f64 / dataset.len() as f64) as f64 * right_entropy;
        let weighted_average = weighted_left_entropy + weighted_right_entropy;

        let information_gain = total_entropy - weighted_average;

        information_gain
    }
}

impl EntropySelectionMeasure {
    pub fn new() -> EntropySelectionMeasure {
        EntropySelectionMeasure {}
    }
}

pub fn entropy(dataset: ArrayView1<f64>) -> f64 {
    let length = dataset.len();

    let ent: f64 = dataset
        .fold(HashMap::new(), |mut histogram, elem: &f64| {
            let key = *elem as usize;
            *histogram.entry(key).or_insert(0) += 1;
            histogram
        })
        .values()
        .map(|h| *h as f64 / length as f64)
        .map(|ratio| ratio * ratio.log2())
        .sum();

    -1.0 * ent
}

