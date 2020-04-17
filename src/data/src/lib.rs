extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use std::fs::File;

use csv::ReaderBuilder;
use ndarray::{Array, Array1, Array2, azip, s};
use ndarray_csv::Array2Reader;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::isaac64::Isaac64Rng;

pub fn read_headbrain_dataset() -> Array2<f64> {
    let csv = include_str!("../headbrain.csv");

    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(csv.as_bytes());
    let dataset: Array2<f64> = reader.deserialize_array2((237, 4)).unwrap();

    return dataset.into_owned();
}

pub fn read_student() -> Array2<f64> {
    let csv = include_str!("../student.csv");

    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(csv.as_bytes());
    let dataset: Array2<f64> = reader.deserialize_array2((1000, 3)).unwrap();

    return dataset.into_owned();
}

pub fn read_banknote_authentication_dataset() -> (Array2<f64>, Array1<bool>) {
    let csv = include_str!("../data_banknote_authentication.csv");

    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(csv.as_bytes());
    let dataset: Array2<f64> = reader.deserialize_array2((1372, 5)).unwrap();

    let x = dataset.slice(s![.., ..4]);
    let y = dataset.slice(s![.., 4]).map(|x| if *x == 1. { true } else { false });

    return (x.into_owned(), y.into_owned());
}

pub fn read_wine_quality_dataset() -> (Array2<f64>, Array1<f64>) {
    let file = File::open("winequality-white.csv").unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_reader(file);

    let dataset: Array2<f64> = reader.deserialize_array2((4898, 12)).unwrap();

    let x = dataset.slice(s![.., ..11]);
    let y = dataset.slice(s![.., 11]);

    return (x.into_owned(), y.into_owned());
}

pub fn xor_dataset(count: usize) -> (Array2<f64>, Array1<bool>) {
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let x = Array::random_using((count, 2), Uniform::new(0., 1.), &mut rng);
    let mut y = Array1::default(x.nrows());

    azip!((y in &mut y, row in x.genrows()) *y = if (row[0] > 0.5 && row[1] > 0.5) || (row[0] < 0.5 && row[1] < 0.5) {true} else {false});

    return (x.into_owned(), y.into_owned());
}