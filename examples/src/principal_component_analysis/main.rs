use log::*;
use ndarray::{s, Array1, ArrayView1, Axis};

use rune_data::{read_banknote_authentication_dataset, read_iris_dataset};
use rune_metrics::confusion_matrix::ConfusionMatrix;
use rune_model_selection::splitting::train_test_split::train_test_split;
use rune_tree::DecisionTreeClassifier;
use rune_tree::feature_selector::greedy_feature_selector::GreedyFeatureSelector;
use rune_tree::measures::entropy::EntropySelectionMeasure;
use ndarray_type_conversion::MapTypeExt;
use itertools::Itertools;
use rune_decomposition::principal_component_analysis::PrincipalComponentAnalysis;
use rune_preprocessing::standard_scaler::StandardScaler;

fn main() {
    env_logger::init();

    let df = read_iris_dataset().unwrap();

    // info!("df.map_type::<f64>(): {:?}", df.map_type::<f64>());

    let x = df.slice(s![.., ..4]).map_type::<f64>();
    let y: Array1<String> = df.slice(s![.., 4]).map_type::<String>();

    let scaler = StandardScaler::new();

    let transformed = scaler.fit(x.view()).transform(x.view());
    info!("transformed: {:?}", transformed);
    info!("transformed mean: {:?}", transformed.mean_axis(Axis(0)));
    info!("transformed std: {:?}", transformed.std_axis(Axis(0), 1.));

    let pca = PrincipalComponentAnalysis::new();
    let x = pca.fit(transformed.view());

    // pca.fit(x.view());
}