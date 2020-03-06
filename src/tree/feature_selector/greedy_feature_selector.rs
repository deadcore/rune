use std::fmt::Debug;
use std::hash::Hash;

use log::*;
use ndarray::{ArrayView1, ArrayView2};

use crate::measures::SelectionMeasure;
use crate::tree::feature_selector::{FeatureSelector, SplitResult};

#[derive(Debug)]
pub struct GreedyFeatureSelector<SM: SelectionMeasure> {
    selection_measure: SM,
}

impl<SM: SelectionMeasure + Debug> GreedyFeatureSelector<SM> {
    pub fn new(selection_measure: SM) -> Self {
        GreedyFeatureSelector {
            selection_measure
        }
    }

    fn split_by_value(&self, x: ArrayView1<f64>, value: f64) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for i in 0..x.len() {
            if x[i] < value {
                left.push(i);
            } else {
                right.push(i);
            }
        }

        return (left, right);
    }
}

impl<SM: SelectionMeasure + Debug> FeatureSelector for GreedyFeatureSelector<SM> {
    fn apply<T: Copy + Eq + Hash>(&self, x: ArrayView2<f64>, y: ArrayView1<T>) -> SplitResult {
        let rows = x.nrows();

        let mut best_score = -1.;
        let mut best_split_value = 0.;
        let mut best_split_column = 0;
        let mut best_left_indexes: Vec<usize> = vec![];
        let mut best_right_indexes: Vec<usize> = vec![];

        for column_index in 0..x.ncols() {
            let columns = x.column(column_index);
            for row_index in 0..rows {
                let split_value = columns[row_index];

                let (left_indexes, right_indexes) = self.split_by_value(columns, split_value);

                let entropy = self.selection_measure.apply(y, left_indexes.as_ref(), right_indexes.as_ref());

                debug!("Split: [X{:} < {:.2}] when information gain = {:.5}", column_index, split_value, entropy);

                if entropy > best_score {
                    best_split_value = split_value;
                    best_split_column = column_index;
                    best_score = entropy;
                    best_left_indexes = left_indexes;
                    best_right_indexes = right_indexes;
                    debug!("New best split: [X{:} < {:.2}] when information gain = {:.5}", best_split_column, best_split_value, best_score);
                }
            }
        }

        info!("Found best split: [X{:} < {:.2}] when information gain = {:.5}", best_split_column, best_split_value, best_score);

        (
            best_left_indexes,
            best_right_indexes,
            best_split_value,
            best_split_column
        )
    }
}