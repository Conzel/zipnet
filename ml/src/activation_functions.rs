//! This module provides the necessary activation functions for our neural networks,
//! namely Relu, Generalized Divisive Normalization (GDN) and it's inverse.
//!
//! All activation functions are exposed as a layer as well as a free function
use ndarray::*;

use crate::{models::InternalDataRepresentation, ImagePrecision, WeightPrecision};

/// Implementation base for GDN, leveraging that we can use almost the same implementation for GDN and iGDN.
/// The explicit functions (gdn, igdn) are publicly exported to serve a nicer interface.
fn gdn_base(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
    params: GdnParameters,
    inverse: bool,
) -> InternalDataRepresentation {
    let num_channels = x.len_of(Axis(0));
    let height = x.len_of(Axis(1));
    let width = x.len_of(Axis(2));

    let mut z: Array3<ImagePrecision> = Array::zeros((num_channels, height, width));

    for i in 0..num_channels {
        let x_i = x.slice(s![i, .., ..]);

        let mut weighted_x_sum: Array2<ImagePrecision> = Array::zeros((height, width));
        for j in 0..num_channels {
            let x_j = x.slice(s![j, .., ..]);
            // TODO: Should we run into some performance problems, this is a bit bad,
            // since it copies the array in a loop...
            match params {
                GdnParameters::New => {
                    weighted_x_sum = weighted_x_sum + gamma[[i, j]] * x_j.mapv(|a| a.abs())
                }
                GdnParameters::Old => {
                    weighted_x_sum = weighted_x_sum + gamma[[i, j]] * x_j.mapv(|a| a.powi(2))
                }
            }
        }

        // normalization before applying the epsilon parameter in the exponent
        let normalization_pre_epsilon = beta[i] + weighted_x_sum;

        let normalization = match params {
            GdnParameters::New => normalization_pre_epsilon,
            GdnParameters::Old => normalization_pre_epsilon.mapv(|a| a.sqrt()),
        };

        let z_i = if inverse {
            &x_i * normalization
        } else {
            &x_i / normalization
        };

        // TODO: Same thing here, a lot of unnecessary assignments :(
        for k in 0..width {
            for l in 0..height {
                z[[i, l, k]] = z_i[[l, k]];
            }
        }
    }
    z
}

/// Sensible parameter setting for GDN/iGDN.
/// Old: alpha = 2, epsilon = 0.5
/// New: alpha = 1, epsilon = 1
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GdnParameters {
    Old,
    New,
}

/// Generalized Divisive Normalization activation function.
/// Refer to Ballé et al., 2016, arXiv:1511.06281v4, https://www.cns.nyu.edu/pub/lcv/balle16a-reprint.pdf
/// for a more in-depth explanation.
/// Regarding the parameter choice:
/// alpha and epsilon to 1 each, as elaborated in https://arxiv.org/abs/1912.08771
/// is the most efficient choice.
///
/// The tensorflow_compression version (1.3) which we use however prevents us
/// from choosing the values of the activation function and fixes them to alpha=2,
/// epsilon = 0.5.
///
/// We pass an enum to detect the parameter choice (old: 2 and 0.5, new: 1 and 1).
/// Other values of epsilon and alpha are not really used so we don't implement them
/// (this allows us to be a bit more efficient with squaring and taking roots).
///
/// i and j here indicate channel parameters (so the different channels in the image influence each other
/// in the activation)
///
/// We expect the data x to be passed in Pytorch layout (channels, height, width).
/// Gamma shall have the form (input channels, output channels)
pub fn gdn(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
    params: GdnParameters,
) -> InternalDataRepresentation {
    gdn_base(x, beta, gamma, params, false)
}

/// Inverse Generalized Divisive Normaliazion, computed by the fix-point method mentioned in
/// Ballé et al., 2016, arXiv:1511.06281v4, https://www.cns.nyu.edu/pub/lcv/balle16a-reprint.pdf
/// We only compute one step of the fixed point iteration (should be sufficient for our use cases,
/// and is in line with the architecture described in Minnen et al, 2018 https://arxiv.org/pdf/1809.02736.pdf)
///
/// Implementation should conform to the tfc implementation for interoperability with trained python layers.
/// Source code: https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py#L31-L461
/// We essentially just replace the division operation with a multiplication operation in the normalization calculation,
/// leveraging the fact that we only perform one step of iteration.
///
/// We expect the data x to be passed in Pytorch layout (channels, height, width).
/// Gamma shall have the form (input channels, output channels)
pub fn igdn(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
    params: GdnParameters,
) -> InternalDataRepresentation {
    gdn_base(x, beta, gamma, params, true)
}

/// Leaky relu implementation
#[allow(dead_code)]
pub fn leaky_relu<D: Dimension>(data: &Array<ImagePrecision, D>) -> Array<ImagePrecision, D> {
    data.mapv(|x| if x > 0. { x } else { 0.01 * x })
}

/// Relu implementation
#[allow(dead_code)]
pub fn relu<D: Dimension>(data: &Array<ImagePrecision, D>) -> Array<ImagePrecision, D> {
    data.mapv(|x| if x > 0. { x } else { 0. })
}

/// Implementation of GDN as a layer. Refer to the documentation of the free GDN function
/// for more info.
pub struct GdnLayer {
    beta: Array1<WeightPrecision>,
    gamma: Array2<WeightPrecision>,
}

impl GdnLayer {
    pub fn new(beta: Array1<WeightPrecision>, gamma: Array2<WeightPrecision>) -> Self {
        Self { beta, gamma }
    }

    /// Performs GDN on the input with the layer parameters.
    /// We fix the parameter choice to alpha=2, epsilon=0.5.
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        gdn(x, &self.beta, &self.gamma, GdnParameters::New)
    }
}

/// Implementation of iGDN as a layer. Refer to the documentation of the free iGDN function
/// for more info.
pub struct IgdnLayer {
    beta: Array1<WeightPrecision>,
    gamma: Array2<WeightPrecision>,
}

impl IgdnLayer {
    pub fn new(beta: Array1<WeightPrecision>, gamma: Array2<WeightPrecision>) -> Self {
        Self { beta, gamma }
    }

    /// Performs iGDN on the input with the layer parameters.
    /// We fix the parameter choice to alpha=2, epsilon=0.5.
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        igdn(x, &self.beta, &self.gamma, GdnParameters::Old)
    }
}

/// Relu implementation.
pub struct ReluLayer {}

impl ReluLayer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        x.map(|a| a.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn_new() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        // Result is taken from tensorflow_compression implementation
        // https://github.com/tensorflow/compression/blob/master/docs/api_docs/python/tfc/GDN.md
        let res = array![[[0., 0.5], [0., 0.33333333]], [[0., 0.], [0., 0.5]]];
        assert_eq!(gdn(&input, &beta, &gamma, GdnParameters::New), res);
    }

    #[test]
    fn test_gdn_old() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        // Result is taken from old tensorflow_compression implementation
        // https://github.com/tensorflow/compression/blob/e1e08a2c62e4d08b93c6bf4008c8a123fc17b2a0/tensorflow_compression/python/layers/gdn.py#L171
        let res = array![
            [[0., 0.70710677], [0., 0.57735026]],
            [[0., 0.], [0., 0.70710677]]
        ];
        assert_eq!(gdn(&input, &beta, &gamma, GdnParameters::Old), res);
    }

    #[test]
    fn test_igdn_new() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        let res_i = array![[[0., 2.], [0., 3.]], [[0., 0.], [0., 2.]]];

        assert_eq!(igdn(&input, &beta, &gamma, GdnParameters::New), res_i);
    }

    #[test]
    fn test_igdn_old() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        let res_i = array![
            [[0., 1.4142135], [0., 1.7320508]],
            [[0., 0.], [0., 1.4142135]]
        ];

        assert_eq!(igdn(&input, &beta, &gamma, GdnParameters::Old), res_i);
    }

    #[test]
    fn test_relu() {
        let x = Array::from_shape_vec((1, 2, 2), vec![1., -2., 3., -4.]).unwrap();
        let out = Array::from_shape_vec((1, 2, 2), vec![1., 0., 3., 0.]).unwrap();
        let relu_layer = ReluLayer::new();
        assert_eq!(relu_layer.activate(&x), out);
    }
}
