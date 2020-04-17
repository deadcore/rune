use log::*;

use ndarray::s;
use rune_data::read_headbrain_dataset;
use rune_model_selection::splitting::train_test_split::train_test_split;
use rune_linear::linear_regression::LinearRegressionRegressor;
use rune_metrics::regression::root_mean_squared_error::root_mean_squared_error;
use rune_metrics::regression::r2::r2;

fn main() {
    env_logger::init();

    let df = read_headbrain_dataset().unwrap();

    let x = df.slice(s![.., ..3]);
    let y = df.slice(s![.., 3]);

    let (x_t_train, x_t_test, y_train, y_test) = train_test_split(x.view(), y.view(), 0.8);

    let x_train = x_t_train.column(2);
    let x_test = x_t_test.column(2);

    info!("x_train: {:?}", x_train);
    info!("x_test: {:?}", x_test);
    info!("y_train: {:?}", y_train);
    info!("y_test: {:?}", y_test);

    let classifier = LinearRegressionRegressor::new();

    info!("Classifier: {:#?}", classifier);

    let model = classifier.fit(x_train.view(), y_train.view());

    info!("trained model: {:#?}", model);

    let y_pred = model.predict(x_test.view());
    info!("Result from test set {:?}", y_pred);

    info!("rmse: {:}", root_mean_squared_error(y_test.view(), y_pred.view()));
    info!("r2: {:}", r2(y_test.view(), y_pred.view()));
}