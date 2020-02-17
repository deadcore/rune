extern crate ndarray;

use std::collections::HashMap;
use std::fmt::Debug;

use log::*;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::measures::entropy::entropy;
use crate::measures::SelectionMeasure;

use self::ndarray::ArrayView;

pub mod feature_selector;

trait FeatureSelector {
    fn apply(&self, x: ArrayView2<f64>);
}

#[derive(Debug)]
pub struct DecisionTreeClassifier<T: SelectionMeasure> {
    max_depth: u32,
    min_size: usize,
    selection_measure: T,
}

#[derive(Debug)]
enum DecisionTreeNode {
    Interior {
        feature: usize,
        threshold: f64,
        left: Box<DecisionTreeNode>,
        right: Box<DecisionTreeNode>,
    },
    Leaf {
        probability: f64,
    },
}

impl DecisionTreeNode {
    fn new_interior(
        feature: usize,
        threshold: f64,
        left: DecisionTreeNode,
        right: DecisionTreeNode,
    ) -> DecisionTreeNode {
        DecisionTreeNode::Interior {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_leaf_node(y: ArrayView1<f64>) -> DecisionTreeNode {
        let distribution = histogram(y);

        let (key, _) = distribution
            .iter()
            .max_by_key(|&(_, value)| {
                value
            }).unwrap();

        DecisionTreeNode::Leaf { probability: *key as f64 }
    }

    pub fn predict(&self, x: ArrayView1<f64>) -> f64 {
        return match *self {
            DecisionTreeNode::Interior { feature, threshold, ref left, ref right } => {
                if x[feature] < threshold {
                    left.predict(x)
                } else {
                    right.predict(x)
                }
            }
            DecisionTreeNode::Leaf { probability } => { return probability; }
        };
    }
}

#[derive(Debug)]
pub struct DecisionTreeModel {
    tree: DecisionTreeNode
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

impl<SM: SelectionMeasure + Debug> DecisionTreeClassifier<SM> {
    pub fn new(max_depth: u32, min_size: usize, selection_measure: SM) -> DecisionTreeClassifier<SM> {
        DecisionTreeClassifier {
            max_depth,
            min_size,
            selection_measure,
        }
    }

    pub fn fit(&self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> DecisionTreeModel {
        DecisionTreeModel {
            tree: self.build_tree(x, y, 0)
        }
    }

    fn build_tree(&self, x: ArrayView2<f64>, y: ArrayView1<f64>, depth: u32) -> DecisionTreeNode {
        let current_entropy = entropy(y);
        info!("Current entropy of split: {:.5}", current_entropy);

        if y.len() <= self.min_size || depth > self.max_depth || current_entropy == 0. {
            info!("Terminating branch with a leaf");
            return DecisionTreeNode::new_leaf_node(y);
        }

        let (left_indexes,
            right_indexes,
            threshold,
            feature) = self.determine_optimal_split_point(x, y);


        let left_y = y.select(Axis(0), left_indexes.as_ref());
        info!("Current depth of: {:} and drafting left side of node", depth);
        let left = self.build_tree(x.select(Axis(0), left_indexes.as_ref()).view(), left_y.view(), depth + 1);

        let right_y = y.select(Axis(0), right_indexes.as_ref());
        info!("Current depth of: {:} and drafting right side of node", depth);
        let right = self.build_tree(x.select(Axis(0), right_indexes.as_ref()).view(), right_y.view(), depth + 1);

        return DecisionTreeNode::new_interior(
            feature,
            threshold,
            left,
            right,
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

fn histogram(ds: ArrayView1<f64>) -> HashMap<usize, usize> {
    ds.fold(HashMap::new(), |mut histogram, elem: &f64| {
        let key = *elem as usize;
        *histogram.entry(key).or_insert(0) += 1;
        histogram
    })
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