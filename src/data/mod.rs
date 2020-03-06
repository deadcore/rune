extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use std::fs::File;

use csv::ReaderBuilder;
use ndarray::{array, Array, Array1, Array2, azip, s};
use ndarray_csv::Array2Reader;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

pub fn read_headbrain_dataset() -> Array2<f64> {
    let file = File::open("headbrain.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let dataset: Array2<f64> = reader.deserialize_array2((237, 4)).unwrap();

    return dataset.into_owned();
}

pub fn read_banknote_authentication_dataset() -> (Array2<f64>, Array1<bool>) {
    let file = File::open("data_banknote_authentication.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let dataset: Array2<f64> = reader.deserialize_array2((1372, 5)).unwrap();

    let x = dataset.slice(s![.., ..4]);
    let y = dataset.slice(s![.., 4]).map(|x| if *x == 1. { true } else { false });

    return (x.into_owned(), y.into_owned());
}

pub fn read_wine_quality_dataset() -> (Array2<f64>, Array1<u8>) {
    let file = File::open("winequality-white.csv").unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_reader(file);

    let dataset: Array2<f64> = reader.deserialize_array2((4898, 12)).unwrap();

    let x = dataset.slice(s![.., ..11]);
    let y = dataset.slice(s![.., 11]).map(|x| *x as u8);

    return (x.into_owned(), y.into_owned());
}

pub fn xor_dataset(count: usize) -> (Array2<f64>, Array1<bool>) {
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let x = Array::random_using((count, 2), Uniform::new(0., 1.), &mut rng);
    let mut y = Array1::default(x.nrows());

    azip!((y in &mut y, row in x.genrows()) *y = if (row[0] > 0.5 && row[1] > 0.5) || (row[0] < 0.5 && row[1] < 0.5) {true} else {false});

    return (x.into_owned(), y.into_owned());
}

pub fn read_static_dataset() -> (Array2<f64>, Array1<bool>) {
    let x = array![
        [80., 20.],
        [66., 32.],
        [43., 12.],
        [82., 28.],
        [65., 32.],
        [42., 35.],
        [70., 39.,],
        [81., 45.,],
        [69., 12.],
    ];

    let y = array![
        false,
        true,
        true,
        true,
        false,
        false,
        true,
        false,
        true,
        false,
    ];

    return (x, y);
}