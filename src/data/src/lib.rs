use csv::ReaderBuilder;
use ndarray::{Array, Array1, Array2, azip, array};
use ndarray_csv::{Array2Reader, ReadError};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ndarray_heterogeneous::Scalar;
use rand_isaac::isaac64::Isaac64Rng;

pub fn read_static_dataset() -> Array2<f64> {
    return array![
    [1., 2., 3.],
    [2., 1., 3.]
    ];
}

pub fn read_iris_dataset() -> Result<Array2<Scalar>, ndarray_csv::ReadError> {
    let csv = include_str!("../iris.csv");

    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(csv.as_bytes());
    let dataset: Array2<Scalar> = reader.deserialize_array2_dynamic()?;

    // let r: Array2<f64> = dataset.mapv(|v| f(v));

    // fn f(s: String) -> f64 {
    //     s.parse().unwrap()
    // }

    Ok(dataset.into_owned())
}

pub fn read_headbrain_dataset() -> Result<Array2<f64>, ndarray_csv::ReadError> {
    let csv = include_str!("../headbrain.csv");

    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(csv.as_bytes());
    let dataset: Array2<f64> = reader.deserialize_array2((237, 4))?;

    Ok(dataset.into_owned())
}

pub fn read_student() -> Result<Array2<f64>, ReadError> {
    let csv = include_str!("../student.csv");

    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(csv.as_bytes());
    let dataset: Array2<f64> = reader.deserialize_array2((1000, 3))?;

    Ok(dataset.into_owned())
}

pub fn read_banknote_authentication_dataset() -> Result<Array2<f64>, ReadError> {
    let csv = include_str!("../data_banknote_authentication.csv");

    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(csv.as_bytes());
    let dataset: Array2<f64> = reader.deserialize_array2((1372, 5))?;

    Ok(dataset)
}

pub fn read_wine_quality_dataset() -> Result<Array2<f64>, ReadError> {
    let csv = include_str!("../winequality-white.csv");

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b';')
        .from_reader(csv.as_bytes());

    let dataset: Array2<f64> = reader.deserialize_array2((4898, 12))?;

    Ok(dataset.into_owned())
}

pub fn xor_dataset(count: usize) -> (Array2<f64>, Array1<bool>) {
    let mut rng = Isaac64Rng::seed_from_u64(42);

    let x = Array::random_using((count, 2), Uniform::new(0., 1.), &mut rng);
    let mut y = Array1::default(x.nrows());

    azip!((y in &mut y, row in x.genrows()) *y = if (row[0] > 0.5 && row[1] > 0.5) || (row[0] < 0.5 && row[1] < 0.5) {true} else {false});

    return (x.into_owned(), y.into_owned());
}