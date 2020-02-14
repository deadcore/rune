extern crate ndarray;

use std::collections::HashMap;

use log::*;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::measures::entropy::EntropySelectionMeasure;
use crate::measures::SelectionMeasure;
use std::fmt::Debug;

#[derive(Debug)]
pub struct DecisionTreeClassifier<T: SelectionMeasure> {
    max_depth: u32,
    min_size: u32,
    selection_measure: T,
}

trait DecisionTreeNode {
    fn predict(&self, x: ArrayView1<f64>) -> f64;
}

#[derive(PartialEq)]
struct TerminalNode {
    prediction: usize
}

impl TerminalNode {
    fn new(value: usize) -> TerminalNode {
        TerminalNode {
            prediction: value
        }
    }

    fn from_labels(y: ArrayView1<f64>) -> TerminalNode {
        let distribution = y
            .fold(HashMap::new(), |mut histogram, elem: &f64| {
                let key = *elem as usize;
                *histogram.entry(key).or_insert(0) += 1;
                histogram
            });

        let (key, _) = distribution
            .iter()
            .max_by_key(|&(_, value)| {
                value
            }).unwrap();

        TerminalNode::new(*key)
    }
}

impl DecisionTreeNode for TerminalNode {
    fn predict(&self, _: ArrayView1<f64>) -> f64 {
        return self.prediction as f64;
    }
}

struct InteriorNode {
    left: Box<dyn DecisionTreeNode>,
    right: Box<dyn DecisionTreeNode>,
    feature: usize,
    threshold: f64,
}

impl InteriorNode {
    fn new(
        left: Box<dyn DecisionTreeNode>,
        right: Box<dyn DecisionTreeNode>,
        feature: usize,
        threshold: f64,
    ) -> InteriorNode {
        InteriorNode {
            left,
            right,
            feature,
            threshold,
        }
    }
}

pub struct DecisionTreeModel {
    tree: Box<dyn DecisionTreeNode>
}

impl DecisionTreeModel {
    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut results = Array1::<f64>::zeros(x.nrows());

        for row_index in 0..x.nrows() {
            let row = x.row(row_index);
            let v = self.tree.predict(row);
            results[[row_index]] = v;
        }

        return results;
    }
}

impl DecisionTreeNode for InteriorNode {
    fn predict(&self, x: ArrayView1<f64>) -> f64 {
        if x[self.feature] < self.threshold {
            self.left.predict(x)
        } else {
            self.right.predict(x)
        }
    }
}

impl <SM: SelectionMeasure + Debug> DecisionTreeClassifier<SM> {
    pub fn new(max_depth: u32, min_size: u32, selection_measure: SM) -> DecisionTreeClassifier<SM> {
        DecisionTreeClassifier {
            max_depth,
            min_size,
            selection_measure,
        }
    }

    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> DecisionTreeModel {
        DecisionTreeModel {
            tree: self.build_tree(x, y, self.max_depth)
        }
    }

    fn build_tree(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, depth: u32) -> Box<dyn DecisionTreeNode> {
        info!("Current depth of: {:}", depth);

        let (left_indexes,
            right_indexes,
            threshold,
            feature) = self.determine_optimal_split_point(x, y);

        let left_y = y.select(Axis(0), left_indexes.as_ref());
        let right_y = y.select(Axis(0), right_indexes.as_ref());

        if depth <= 1 {
            let left = Box::new(TerminalNode::from_labels(left_y.view()));
            let right = Box::new(TerminalNode::from_labels(right_y.view()));

            return Box::new(
                InteriorNode::new(
                    left,
                    right,
                    feature,
                    threshold,
                )
            );
        }

        if left_y.is_empty() || right_y.is_empty() {
            if left_y.is_empty() {
                return Box::new(TerminalNode::from_labels(right_y.view()));
            }
            return Box::new(TerminalNode::from_labels(left_y.view()));
        }

        let left = self.build_tree(x.select(Axis(0), left_indexes.as_ref()).view(), left_y.view(), depth - 1);
        let right = self.build_tree(x.select(Axis(0), right_indexes.as_ref()).view(), right_y.view(), depth - 1);

        return Box::new(
            InteriorNode::new(
                left,
                right,
                feature,
                threshold,
            )
        );
    }

    fn determine_optimal_split_point(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> (Vec<usize>, Vec<usize>, f64, usize) {
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

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2, s};

    use crate::tree::TerminalNode;

    #[test]
    fn test_terminal_node_takes_majority() {
        let groups = array![1.,0.,1.];

        let terminal_node = TerminalNode::from_labels(groups.view());
        assert!(terminal_node == TerminalNode::new(1))
    }
}