use crate::{ImagePrecision, WeightPrecision};
use ndarray::*;

/// Rust implementation of a feed forward layer.
/// The weight matrix shall have dimension (in that order)
/// input units x output units (to comply with the order in which pytorch weights are saved).
pub struct FeedforwardLayer {
    weights: Array2<WeightPrecision>,
    input_dimension: usize,
    output_dimension: usize,
}

impl FeedforwardLayer {
    pub fn new(weights: Array2<WeightPrecision>) {
        todo!()
    }

    pub fn forward_pass<'a, V>(data: V) -> Array1<ImagePrecision>
    where
        V: AsArray<'a, ImagePrecision, Ix1>,
    {
        let data_arr: ArrayView1<ImagePrecision> = data.into();
        todo!()
    }
}
