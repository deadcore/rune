use ndarray::{Array, Axis, RemoveAxis, ShapeBuilder, Data};
use ndarray::{ArrayBase, DataOwned, Dimension};
use std::convert::{TryInto, TryFrom};


pub trait MapTypeExt<A, S, D>
    where
        S: Data<Elem=A>,
        D: Dimension,
        A: Clone
{
    fn map_type<T: From<A>>(&self) -> Array<T, D>;
}

impl<A, S, D> MapTypeExt<A, S, D> for ArrayBase<S, D> where
    S: Data<Elem=A>,
    D: Dimension,
    A: Clone
{
    fn map_type<T: From<A>>(&self) -> Array<T, D> {
        let z: Array<T, D> = self.mapv(|v| T::from(v));
        z
    }
}

// fn main() {
//     env_logger::init();
//
//     let t = Scalar::F64(1.0);
//     let f = Scalar::I64(0);
//
//     info!("t: {:?}", t);
//     info!("f: {:?}", f);
//
//     info!("t_bool: {:?}", t.unwrap_as::<bool>());
//     info!("f_bool: {:?}", f.unwrap_as::<bool>());
//
//     let df = read_heterogeneous_data();
//
//     info!("df: {:?}", df);
//     info!("df.map_type::<f64>(): {:?}", df.map_type::<f64>());
// }