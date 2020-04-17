use std::collections::HashMap;
use std::hash::Hash;

use ndarray::ArrayView1;

pub fn histogram<T: Eq + Hash + Copy>(ds: ArrayView1<T>) -> HashMap<T, usize> {
    ds.fold(HashMap::new(), |mut histogram, &elem| {
        histogram.entry(elem)
                  .and_modify(|e| { *e += 1 })
                  .or_insert(1);
        histogram
    })
}
