use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::prelude::*;

pub fn train_test_split<X: Copy, Y: Copy>(x: ArrayView2<X>, y: ArrayView1<Y>, ratio: f32) -> (Array2<X>, Array2<X>, Array1<Y>, Array1<Y>) {
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