use crate::tree::FeatureSelector;

use super::ndarray::{ArrayBase, ArrayView, ArrayView2, ViewRepr};

struct GreedyFeatureSelector {}

impl FeatureSelector for GreedyFeatureSelector {
    fn apply(&self, x: ArrayView2<f64>) {
        unimplemented!()
    }
}

struct PercentileFeatureSelector {}

