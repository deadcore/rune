extern crate ndarray;

use log::*;
use ndarray::array;

use rust_decision_tree::data::{read_banknote_authentication_dataset, read_static_dataset};
use rust_decision_tree::measures::entropy::EntropySelectionMeasure;
use rust_decision_tree::metrics::{false_positives, precision, true_positives, recall, f1, false_negatives, true_negatives};
use rust_decision_tree::model_selection::splitting::train_test_split;
use rust_decision_tree::tree::DecisionTreeClassifier;

fn main() {
    env_logger::init();

    let (x, y) = read_banknote_authentication_dataset();

    let (x_train, x_test, y_train, y_test) = train_test_split(x.view(), y.view(), 0.8);

    info!("x_train: {:?}", x_train);
    info!("x_test: {:?}", x_test);
    info!("y_train: {:?}", y_train);
    info!("y_test: {:?}", y_test);

    let decision_tree = DecisionTreeClassifier::new(
        10,
        3,
        EntropySelectionMeasure::new(),
    );

    info!("{:?}", decision_tree);

    let model = decision_tree.fit(x_train.view(), y_train.view());

    info!("trained model: {:#?}", model);


    let y_pred = model.predict(x_test.view());
    info!("Result from test set {:?}", y_pred);

    info!("Count: {:}", y_pred.len());
    info!("TP: {:}", true_positives(y_test.view(), y_pred.view()));
    info!("FP: {:}", false_positives(y_test.view(), y_pred.view()));
    info!("FN: {:}", false_negatives(y_test.view(), y_pred.view()));
    info!("TN: {:}", true_negatives(y_test.view(), y_pred.view()));
    info!("Precision: {:.5}", precision(y_test.view(), y_pred.view()));
    info!("Recall: {:.5}", recall(y_test.view(), y_pred.view()));
    info!("F1: {:.5}", f1(y_test.view(), y_pred.view()));

}