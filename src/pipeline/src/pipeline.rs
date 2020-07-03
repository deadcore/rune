use ndarray::{ArrayView2, Array2, ArrayView1};
use std::marker::PhantomData;

// pub struct Pipeline<In, Out, F, Tf> {
//     _in: PhantomData<In>,
//     _out: PhantomData<Out>,
//     _tf: PhantomData<Tf>,
//
//     f: F,
// }

pub trait Transformer<In, Out> {
    fn transform(&self, x: In) -> Out;
}

pub trait Fit<In, Out> {
    fn fit(&self, x: In, y: ArrayView1<bool>) -> Out;
}

pub struct ComposedTransform<In, F1Output, Out, F1Transformer, F2Transformer> {
    _in: PhantomData<In>,
    _out: PhantomData<Out>,

    _F1Transformer: PhantomData<F1Transformer>,
    _F1Output: PhantomData<F1Output>,
    _F2Transformer: PhantomData<F2Transformer>,
    t1: F1Transformer,
    t2: F2Transformer,
}

impl<In, F1Output, Out, F1Transformer, F2Transformer> Transformer<In, Out> for ComposedTransform<In, F1Output, Out, F1Transformer, F2Transformer>
    where
        F1Transformer: Transformer<In, F1Output>,
        F2Transformer: Transformer<F1Output, Out>,
        In: Copy {
    fn transform(&self, x: In) -> Out {
        let t1 = self.t1.transform(x);
        let t2 = self.t2.transform(t1);
        t2
    }
}

pub struct ComposedFit<F1, F2, In, Out, F1Transformer, F1Output, F2Transformer> {
    _in: PhantomData<In>,
    _out: PhantomData<Out>,

    _F1Transformer: PhantomData<F1Transformer>,
    _F1Output: PhantomData<F1Output>,
    _F2Transformer: PhantomData<F2Transformer>,

    f1: F1,
    f2: F2,
}

impl<F1, F2, In, Out, F1Transformer, F1Output, F2Transformer> ComposedFit<F1, F2, In, Out, F1Transformer, F1Output, F2Transformer>
    where
        F1: Fit<In, F1Transformer>,
        F1Transformer: Transformer<In, F1Output>,
        F2: Fit<F1Output, F2Transformer>,
        F2Transformer: Transformer<F1Output, Out>,
        In: Copy {
    pub fn compose(f1: F1, f2: F2) -> Self {
        ComposedFit { _in: PhantomData, _out: PhantomData, _F1Transformer: PhantomData, _F1Output: PhantomData, _F2Transformer: PhantomData, f1, f2 }
    }
}

impl<F1, F2, In, Out, F1Transformer, F1Output, F2Transformer> Fit<In, ComposedTransform<In, F1Output, Out, F1Transformer, F2Transformer>> for ComposedFit<F1, F2, In, Out, F1Transformer, F1Output, F2Transformer>
    where
        F1: Fit<In, F1Transformer>,
        F1Transformer: Transformer<In, F1Output>,
        F2: Fit<F1Output, F2Transformer>,
        F2Transformer: Transformer<F1Output, Out>,
        In: Copy {
    fn fit(&self, x: In, y: ArrayView1<bool>) -> ComposedTransform<In, F1Output, Out, F1Transformer, F2Transformer> {
        let t1 = self.f1.fit(x, y);
        let t2 = self.f2.fit(t1.transform(x), y);
        ComposedTransform { _in: PhantomData, _out: PhantomData, _F1Transformer: PhantomData, _F1Output: PhantomData, _F2Transformer: PhantomData, t1, t2 }
    }
}

//
// impl<In, Out, F, Tf> Pipeline<In, Out, F, Tf> where Tf: Transformer<In, Out>, F: Fit<In, Tf> {
//     pub fn new(f: F) -> Pipeline<In, Out, F, Tf> {
//         Pipeline { _in: PhantomData, _out: PhantomData, _tf: PhantomData, f }
//     }
//
//     pub fn then<IIn, NFit, Nout, NTf>(&self, f: NFit) -> Pipeline<In, Nout, NFit, NTf> where NTf: Transformer<IIn, Nout>, F: Fit<Out, Tf> {
//         let t = Pipeline { _in: PhantomData, _out: PhantomData, _tf: PhantomData, f: f };
//     }
//
//     // pub fn fit(&self, x: In) {
//     //     self.f.fit(x)
//     // }
// }