use log::*;
use ndarray::s;

use rune_data::{read_student, read_banknote_authentication_dataset};
use rune_linear::multiple_linear_regression::MultipleLinearRegression;
use rune_metrics::regression::r2::r2;
use rune_metrics::regression::root_mean_squared_error::root_mean_squared_error;
use rune_model_selection::splitting::train_test_split::train_test_split;
use rune_metrics::confusion_matrix::ConfusionMatrix;

fn main() {
    env_logger::init();

    let df = read_student().unwrap();

    let df = read_banknote_authentication_dataset().unwrap();

    let x = df.slice(s![.., ..4]);
    let y = df.slice(s![.., 4]).map(|x| if *x == 1. { true } else { false });

    let mut cm = ConfusionMatrix::from_labels(y.view());

    let (x_train, x_test, y_train, y_test) = train_test_split(x.view(), y.view(), 0.8);

    info!("x_train: {:?}", x_train);
    info!("x_test: {:?}", x_test);
    info!("y_train: {:?}", y_train);
    info!("y_test: {:?}", y_test);

    let classifier = MultipleLinearRegression::new(
        0.0001,
        10000,
    );

    info!("Classifier: {:#?}", classifier);

    let model = classifier.fit(x_train.view(), y_train.view());

    info!("trained model: {:#?}", model);

    let y_pred = model.predict(x_test.view());
    info!("Result from test set {:?}", y_pred);
    info!("rmse: {:}", root_mean_squared_error(y_test.view(), y_pred.view()));
    info!("r2: {:}", r2(y_test.view(), y_pred.view()));
}