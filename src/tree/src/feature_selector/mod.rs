use std::hash::Hash;
use ndarray::{ArrayView2, ArrayView1};


pub mod greedy_feature_selector;

type IndexSelector = usize;
type IndexSelectors = Vec<IndexSelector>;

type LeftIndexes = IndexSelectors;
type RightIndexes = IndexSelectors;
type SplitThreshold = f64;
type FeatureIndex = IndexSelector;

type SplitResult = (LeftIndexes, RightIndexes, SplitThreshold, FeatureIndex);

pub trait FeatureSelector {
    fn apply<T: Copy + Eq + Hash>(&self, x: ArrayView2<f64>, y: ArrayView1<T>) -> SplitResult;
}