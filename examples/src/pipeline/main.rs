use log::*;
use ndarray::s;

use rune_data::read_banknote_authentication_dataset;
use rune_metrics::confusion_matrix::ConfusionMatrix;
use rune_model_selection::splitting::train_test_split::train_test_split;
use rune_tree::DecisionTreeClassifier;
use rune_tree::feature_selector::greedy_feature_selector::GreedyFeatureSelector;
use rune_tree::measures::entropy::EntropySelectionMeasure;
use rune_preprocessing::standard_scaler::*;
use rune_decomposition::principal_component_analysis::PrincipalComponentAnalysis;
use std::error::Error;
use rune_pipeline::pipeline::{Fit, Transformer, ComposedFit};

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let df = read_banknote_authentication_dataset().unwrap();

    let x = df.slice(s![.., ..4]);
    let y = df.slice(s![.., 4]).map(|x| if *x == 1. { true } else { false });

    let mut cm = ConfusionMatrix::from_labels(y.view());

    let (x_train, x_test, y_train, y_test) = train_test_split(x.view(), y.view(), 0.8);

    let scaler = StandardScaler::new();
    let pca = PrincipalComponentAnalysis::new(1);
    let decision_tree = DecisionTreeClassifier::new(
        2,
        3,
        GreedyFeatureSelector::new(
            EntropySelectionMeasure::new(),
        ),
    );

    let pipeline = ComposedFit::compose(
        scaler,
        pca,
    );
    let pipeline = ComposedFit::compose(
        pipeline,
        decision_tree,
    );

    let model = pipeline.fit(x_train.view(), y_train.view());

    info!("x_test: {:?}", x_test);
    info!("y_test: {:?}", y_test);
    let y_pred = model.transform(x_test.view());
    info!("y_pred: {:?}", y_pred);

    cm.add_all(y_test.view(), y_pred.view());

    info!("Confusion matrix: {:#?}", cm);
    info!("recall:    {:.5}", cm.recall());
    info!("precision: {:.5}", cm.precision());
    info!("f1:        {:.5}", cm.f1());

    Ok(())
}