extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;

use std::fs::File;

use csv::ReaderBuilder;
use ndarray::{array, Array2, s};
use ndarray_csv::Array2Reader;

use self::ndarray::Array1;

pub fn read_banknote_authentication_dataset() -> (Array2<f64>, Array1<f64>) {
    let file = File::open("data_banknote_authentication.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let dataset: Array2<f64> = reader.deserialize_array2((1372, 5)).unwrap();

    let x = dataset.slice(s![.., ..4]);
    let y = dataset.slice(s![.., 4]);

    return (x.into_owned(), y.into_owned());
}

pub fn read_static_dataset() -> (Array2<f64>, Array1<f64>) {
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
        0.,
        1.,
        1.,
        1.,
        0.,
        0.,
        1.,
        0.,
        1.,
        0.,
    ];

    return (x, y);
}