use crate::{ImagePrecision, WeightPrecision};
use ndarray::*;

/// Rust implementation of a feed forward layer.
/// The weight matrix shall have dimension (in that order)
/// input units x output units (to comply with the order in which pytorch weights are saved).
pub struct FullyConnectedLayer {
    weights: Array2<WeightPrecision>,
    bias: Array1<WeightPrecision>,
    input_dimension: usize,
    output_dimension: usize,
}

impl FullyConnectedLayer {
    pub fn new(
        weights: Array2<WeightPrecision>,
        bias: Array1<WeightPrecision>,
    ) -> FullyConnectedLayer {
        let input_dimension = weights.len_of(Axis(0));
        let output_dimension = weights.len_of(Axis(1));
        FullyConnectedLayer {
            weights,
            bias,
            input_dimension,
            output_dimension,
        }
    }

    pub fn forward_pass<'a, V>(self, prev_activations: V) -> Array1<ImagePrecision>
    where
        V: AsArray<'a, ImagePrecision, Ix1>,
    {
        let activations_arr: ArrayView1<ImagePrecision> = prev_activations.into();
        debug_assert_eq!(activations_arr.len_of(Axis(0)), self.input_dimension);
        activations_arr.dot(&self.weights) + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_fc_forward_pass() {
        let weights = array![[0., 1., 0.], [0., 0., 1.]];
        let bias = array![0., 0., 1.];
        let prev_activations = [1., 1.];

        let net = FullyConnectedLayer::new(weights, bias);

        assert_eq!(net.forward_pass(&prev_activations), array![0., 1., 2.]);
    }
}
