use std::collections::{HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use itertools::Itertools;
use log::*;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::iter::FromIterator;

#[derive(Debug)]
pub struct ConfusionMatrix<T: Debug + Eq + Hash> {
    labels: HashMap<T, usize>,
    arr: Array2<u64>,
}

impl<T: Eq + Hash + Debug> ConfusionMatrix<T> {
    fn new(labels: HashMap<T, usize>, arr: Array2<u64>) -> Self {
        ConfusionMatrix { labels, arr }
    }

    pub fn from_labels(labels: ArrayView1<T>) -> ConfusionMatrix<T> {


        let itter = labels.iter().unique();

        let x = Array1::from_iter(itter);

        info!("Log: {:?}", x);

        // itertools::assert_equal(itter.unique(), vec!["Hello"]);

        ConfusionMatrix::new(
            HashMap::new(),
            Array2::zeros((0, 0)),
        )
    }

    // pub fn add(&mut self, y_true: T, y_pred: T) {
    //     let x = *self.labels.get(&y_true).unwrap();
    //     let y = *self.labels.get(&y_pred).unwrap();
    //
    //     self.arr[[x, y]] += 1;
    // }
    //
    // pub fn add_all(&mut self, y_true: ArrayView1<T>, y_pred: ArrayView1<T>) {
    //     for (&prediction, &target) in y_pred.iter().zip(y_true.iter()) {
    //         self.add(target, prediction);
    //     }
    // }

    pub fn false_positive(&self) -> Array1<u64> {
        self.arr.sum_axis(Axis(0)) - self.arr.diag()
    }

    pub fn false_negative(&self) -> Array1<u64> {
        self.arr.sum_axis(Axis(1)) - self.arr.diag()
    }

    pub fn true_positive(&self) -> Array1<u64> {
        self.arr.diag().into_owned()
    }

    pub fn recall(&self) -> Array1<f64> {
        self.true_positive().mapv(|x| x as f64) / (self.false_negative().mapv(|x| x as f64) + self.true_positive().mapv(|x| x as f64))
    }

    pub fn precision(&self) -> Array1<f64> {
        self.true_positive().mapv(|x| x as f64) / (self.false_positive().mapv(|x| x as f64) + self.true_positive().mapv(|x| x as f64))
    }

    pub fn f1(&self) -> Array1<f64> {
        ((self.precision() * self.recall()) / (self.precision() + self.recall())) * 2.0
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2, s};

    use crate::metrics::{f1, precision, recall};

    #[test]
    fn test_recall() {
        let y_true = array![1.,0.,1.,0.];
        let y_pred = array![1.,0.,0.,1.];

        let rc = recall(y_true.view(), y_pred.view());
        assert_eq!(rc, 0.5)
    }

    #[test]
    fn test_precision() {
        let y_true = array![1.,0.,1.,0.];
        let y_pred = array![1.,0.,0.,1.];

        let rc = precision(y_true.view(), y_pred.view());
        assert_eq!(rc, 0.5)
    }

    #[test]
    fn test_f1() {
        let y_true = array![1.,0.,1.,0.];
        let y_pred = array![1.,0.,0.,1.];

        let rc = f1(y_true.view(), y_pred.view());
        assert_eq!(rc, 0.5)
    }
}