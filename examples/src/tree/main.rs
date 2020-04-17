use log::*;
use rune_data::read_banknote_authentication_dataset;
use rune_metrics::confusion_matrix::ConfusionMatrix;
use rune_model_selection::splitting::train_test_split::train_test_split;
use rune_tree::DecisionTreeClassifier;
use rune_tree::feature_selector::greedy_feature_selector::GreedyFeatureSelector;
use rune_tree::measures::entropy::EntropySelectionMeasure;


fn main() {
    env_logger::init();

    let (x, y) = read_banknote_authentication_dataset();

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