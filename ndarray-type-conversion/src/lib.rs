use ndarray::{Array, Data};
use ndarray::{ArrayBase, Dimension};


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