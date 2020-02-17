use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;

pub fn train_test_split(x: ArrayView2<f64>, y: ArrayView1<f64>, ratio: f32) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();
    let mut left = Vec::new();
    let mut right = Vec::new();

    let mut vec: Vec<usize> = (0..x.nrows()).collect();
    vec.shuffle(&mut thread_rng());


    for idx in vec {
        let n1: f32 = rng.gen();

        if n1 < ratio {
            left.push(idx)
        } else {
            right.push(idx)
        }
    }

    let left_indexes = left.as_slice();
    let right_indexes = right.as_slice();

    return (
        x.select(Axis(0), left_indexes),
        x.select(Axis(0), right_indexes),
        y.select(Axis(0), left_indexes),
        y.select(Axis(0), right_indexes)
    );
}