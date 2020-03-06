extern crate ndarray;

use log::*;

use rust_decision_tree::data::{read_banknote_authentication_dataset, read_static_dataset, read_wine_quality_dataset, xor_dataset};
use rust_decision_tree::measures::entropy::EntropySelectionMeasure;
use rust_decision_tree::metrics::ConfusionMatrix;
use rust_decision_tree::model_selection::splitting::train_test_split;
use rust_decision_tree::tree::DecisionTreeClassifier;
use rust_decision_tree::tree::feature_selector::greedy_feature_selector::GreedyFeatureSelector;
use rust_decision_tree::metrics::confusion_matrix::ConfusionMatrix;

fn main() {
    env_logger::init();

    let (x, y) = read_static_dataset();

    let mut cm = ConfusionMatrix::from_labels(y.view());


    let (x_train, x_test, y_train, y_test) = train_test_split(x.view(), y.view(), 0.8);

    info!("x_train: {:?}", x_train);
    info!("x_test: {:?}", x_test);
    info!("y_train: {:?}", y_train);
    info!("y_test: {:?}", y_test);

    let decision_tree = DecisionTreeClassifier::new(
        4,
        3,
        GreedyFeatureSelector::new(
            EntropySelectionMeasure::new(),
        ),
    );

    info!("Decision tree: {:#?}", decision_tree);

    let model = decision_tree.fit(x_train.view(), y_train.view());

    info!("trained model: {:#?}", model);

    let y_pred = model.predict(x_test.view());
    info!("Result from test set {:?}", y_pred);

    info!("Count: {:}", y_pred.len());

    cm.add_all(y_test.view(), y_pred.view());

    info!("Confusion matrix: {:#?}", cm);
    info!("recall:    {:.5}", cm.recall());
    info!("precision: {:.5}", cm.precision());
    info!("f1:        {:.5}", cm.f1());

}