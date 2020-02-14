extern crate ndarray;

use log::*;
use ndarray::array;

use rust_decision_tree::data::{read_banknote_authentication_dataset, read_static_dataset};
use rust_decision_tree::measures::entropy::EntropySelectionMeasure;
use rust_decision_tree::tree::DecisionTreeClassifier;

fn main() {
    env_logger::init();

    let (x, y) = read_banknote_authentication_dataset();

    let decision_tree = DecisionTreeClassifier::new(
        3,
        3,
        EntropySelectionMeasure::new(),
    );

    info!("{:?}", decision_tree);

    let model = decision_tree.fit(x.view(), y.view());

    info!("trained model: {:#?}", model);

    let test = array![[0.062525,2.9301,-3.5467,-2.6737]];

    let result = model.predict(test.view());

    info!("Result from test of {:} was {:}", test, result);
}