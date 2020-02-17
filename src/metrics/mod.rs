use log::*;
use ndarray::{Array1, ArrayView, ArrayView1, azip};

// F-Measure = (2 * Precision * Recall) / (Precision + Recall)
pub fn f1(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f32 {
    let precision = precision(y_true, y_pred);
    let recall = recall(y_true, y_pred);

    (2. * precision * recall) / (precision + recall)
}

// Recall = TruePositives / (TruePositives + FalseNegatives)
pub fn recall(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f32 {
    true_positives(y_true, y_pred) as f32 / (true_positives(y_true, y_pred) + false_negatives(y_true, y_pred)) as f32
}

// Precision = TruePositives / (TruePositives + FalsePositives)
pub fn precision(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> f32 {
    true_positives(y_true, y_pred) as f32 / (true_positives(y_true, y_pred) + false_positives(y_true, y_pred)) as f32
}

pub fn true_positives(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> u32 {
    let mut totals: Array1<bool> = Array1::default(y_true.len());
    azip!((totals in &mut totals, y_true in y_true, y_pred in y_pred) *totals = *y_true == 1. && *y_pred == 1.);

    totals.fold(0, |count, elem| if *elem { return count + 1; } else { count })
}

pub fn false_positives(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> u32 {
    let mut totals: Array1<bool> = Array1::default(y_true.len());
    azip!((totals in &mut totals, y_true in y_true, y_pred in y_pred) *totals = *y_true == 0. && *y_pred == 1.);

    totals.fold(0, |count, elem| if *elem { return count + 1; } else { count })
}

pub fn false_negatives(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> u32 {
    let mut totals: Array1<bool> = Array1::default(y_true.len());
    azip!((totals in &mut totals, y_true in y_true, y_pred in y_pred) *totals = *y_true == 1. && *y_pred == 0.);

    totals.fold(0, |count, elem| if *elem { return count + 1; } else { count })
}

pub fn true_negatives(y_true: ArrayView1<f64>, y_pred: ArrayView1<f64>) -> u32 {
    let mut totals: Array1<bool> = Array1::default(y_true.len());
    azip!((totals in &mut totals, y_true in y_true, y_pred in y_pred) *totals = *y_true == 0. && *y_pred == 0.);

    totals.fold(0, |count, elem| if *elem { return count + 1; } else { count })
}