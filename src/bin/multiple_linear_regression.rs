use log::*;

use rust_decision_tree::data::{read_banknote_authentication_dataset, read_static_dataset, read_wine_quality_dataset, xor_dataset, read_headbrain_dataset, read_student};
use rust_decision_tree::measures::entropy::EntropySelectionMeasure;
use rust_decision_tree::metrics::root_mean_squared_error::{r2, root_mean_squared_error};
use rust_decision_tree::model_selection::splitting::train_test_split;
use rust_decision_tree::regression::linear_regression::LinearRegressionRegressor;
use rust_decision_tree::tree::DecisionTreeClassifier;
use rust_decision_tree::tree::feature_selector::greedy_feature_selector::GreedyFeatureSelector;

use ndarray::s;
use rust_decision_tree::regression::multiple_linear_regression::MultipleLinearRegression;

fn main() {
    env_logger::init();

    let df = read_student();

    let x = df.slice(s![.., ..2]);
    let y = df.slice(s![.., 2]);

    let (x_train, x_test, y_train, y_test) = train_test_split(x.view(), y.view(), 0.8);

    info!("x_train: {:?}", x_train);
    info!("x_test: {:?}", x_test);
    info!("y_train: {:?}", y_train);
    info!("y_test: {:?}", y_test);

    let classifier = MultipleLinearRegression::new(
        0.0001,
        100000,
    );

    info!("Classifier: {:#?}", classifier);

    let model = classifier.fit(x_train.view(), y_train.view());

    info!("trained model: {:#?}", model);

    let y_pred = model.predict(x_test.view());
    info!("Result from test set {:?}", y_pred);
    info!("rmse: {:}", root_mean_squared_error(y_test.view(), y_pred.view()));
    info!("r2: {:}", r2(y_test.view(), y_pred.view()));
}