pub mod feature_selector;
pub mod measures;
pub mod math;

use std::fmt::Debug;
use std::hash::Hash;

use log::*;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use crate::feature_selector::FeatureSelector;
use crate::math::histogram::histogram;
use crate::measures::entropy::entropy;


#[derive(Debug)]
pub struct DecisionTreeClassifier<FS> {
    max_depth: u32,
    min_size: usize,
    feature_selector: FS,
}

#[derive(Debug)]
enum DecisionTreeNode<T> {
    Interior {
        feature: usize,
        threshold: f64,
        left: Box<DecisionTreeNode<T>>,
        right: Box<DecisionTreeNode<T>>,
    },
    Leaf {
        probability: T,
    },
}

impl<T: Copy + Eq + Hash> DecisionTreeNode<T> {
    fn new_interior(
        feature: usize,
        threshold: f64,
        left: DecisionTreeNode<T>,
        right: DecisionTreeNode<T>,
    ) -> DecisionTreeNode<T> {
        DecisionTreeNode::Interior {
            feature,
            threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn new_leaf_node(y: ArrayView1<T>) -> DecisionTreeNode<T> {
        let distribution = histogram(y);

        let (key, _) = distribution
            .iter()
            .max_by_key(|&(_, value)| {
                value
            }).unwrap();

        DecisionTreeNode::Leaf { probability: *key }
    }

    pub fn predict(&self, x: ArrayView1<f64>) -> T {
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
pub struct DecisionTreeModel<T> {
    tree: DecisionTreeNode<T>
}

impl<T: Eq + Hash + Default + Copy> DecisionTreeModel<T> {
    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<T> {
        let mut results = Array1::<T>::default(x.nrows());

        for row_index in 0..x.nrows() {
            let row = x.row(row_index);
            let v = self.tree.predict(row);
            results[[row_index]] = v;
        }

        return results;
    }
}

impl<FS> DecisionTreeClassifier<FS> where FS: FeatureSelector + Debug {
    pub fn new(max_depth: u32, min_size: usize, feature_selector: FS) -> Self {
        DecisionTreeClassifier {
            max_depth,
            min_size,
            feature_selector,
        }
    }

    pub fn fit<Y: Copy + Hash + Eq>(&self, x: ArrayView2<f64>, y: ArrayView1<Y>) -> DecisionTreeModel<Y> {
        DecisionTreeModel {
            tree: self.build_tree(x, y, 0)
        }
    }

    fn build_tree<Y: Copy + Hash + Eq>(&self, x: ArrayView2<f64>, y: ArrayView1<Y>, depth: u32) -> DecisionTreeNode<Y> {
        let current_entropy = entropy(y);
        info!("Current entropy of split: {:.5}", current_entropy);

        if y.len() <= self.min_size || depth > self.max_depth || current_entropy == 0. {
            info!("Terminating branch with a leaf");
            return DecisionTreeNode::new_leaf_node(y);
        }

        let (left_indexes,
            right_indexes,
            threshold,
            feature) = self.feature_selector.apply(x, y);


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
}