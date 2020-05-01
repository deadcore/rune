use ndarray::{Array, Axis, RemoveAxis, ShapeBuilder};
use ndarray::{ArrayBase, DataOwned, Dimension};
use std::convert::{TryInto, TryFrom};


#[derive(Debug, Copy, Clone)]
pub enum Scalar {
    I64(i64),
    F64(f64),
    BOOL(bool),
}

impl From<Scalar> for f64 {
    fn from(scalar: Scalar) -> Self {
        match scalar {
            Scalar::I64(i) => i as f64,
            Scalar::F64(i) => i,
            Scalar::BOOL(i) => if i { 1. } else { 0. },
        }
    }
}

impl From<Scalar> for bool {
    fn from(scalar: Scalar) -> Self {
        match scalar {
            Scalar::F64(i) if i == 0. => false,
            Scalar::F64(i) if i == 1. => true,
            Scalar::I64(i) if i == 1 => true,
            Scalar::I64(i) if i == 0 => false,
            Scalar::BOOL(i) => i,
            _ => panic!("bang")
        }
    }
}

impl Scalar {
    pub fn unwrap_as<B: From<Scalar>>(self) -> B where Self: Sized {
        B::from(self)
    }
}

pub trait ScalarExt<S, D>
    where
        S: DataOwned<Elem=Scalar>,
        D: Dimension,
{
    fn map_scalar_type<T: From<Scalar>>(&self) -> Array<T, D>
        where
            D: RemoveAxis;
}

impl<S, D> ScalarExt<S, D> for ArrayBase<S, D>
    where
        S: DataOwned<Elem=Scalar>,
        D: Dimension,
{
    fn map_scalar_type<T: From<Scalar>>(&self) -> Array<T, D>
        where
            D: RemoveAxis {
        self.mapv(|v| v.unwrap_as::<T>())
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