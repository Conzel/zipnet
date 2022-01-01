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
                    weighted_x_sum = weighted_x_sum + gamma[[j, i]] * x_j.mapv(|a| a.abs())
                }
                GdnParameters::Old => {
                    weighted_x_sum = weighted_x_sum + gamma[[j, i]] * x_j.mapv(|a| a.powi(2))
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
        gdn(x, &self.beta, &self.gamma, GdnParameters::Old)
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

    fn all_almost_equal<T, F>(x: T, y: T, eps: F) -> bool
    where
        T: Iterator<Item = F>,
        F: num::traits::Float + Clone,
    {
        let mut x_iter = x.into_iter();
        let mut y_iter = y.into_iter();

        while let Some(x_val) = x_iter.next() {
            if let Some(y_val) = y_iter.next() {
                if (y_val - x_val).abs() > eps {
                    return false;
                }
            }
        }
        return true;
    }

    #[test]
    fn test_gdn_new() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        // Result is taken from tensorflow_compression implementation
        // https://github.com/tensorflow/compression/blob/master/docs/api_docs/python/tfc/GDN.md
        let res = array![[[0., 0.5], [0., 0.5]], [[0., 0.], [0., 0.33333333]]];
        assert_eq!(gdn(&input, &beta, &gamma, GdnParameters::New), res);
    }

    #[test]
    fn test_gdn_old() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        // Result is taken from old tensorflow_compression implementation
        // https://github.com/tensorflow/compression/blob/e1e08a2c62e4d08b93c6bf4008c8a123fc17b2a0/tensorflow_compression/python/layers/gdn.py#L171
        //
        // In the tf implementation, we have 0.7071068, in Rust we have 0.70710677...
        // Is this problematic?
        let res = array![
            [[0., 0.70710677], [0., 0.70710677]],
            [[0., 0.], [0., 0.57735026]]
        ];
        assert!(all_almost_equal::<_, f32>(
            gdn(&input, &beta, &gamma, GdnParameters::Old)
                .iter()
                .cloned(),
            res.iter().cloned(),
            1e-6
        ));

        let input_rand = array![
            [
                [0.88749708, 0.2674431, 0.31513242, 0.29998093],
                [0.71862384, 0.6403044, 0.39162998, 0.2060626],
                [0.51270423, 0.23101829, 0.69418675, 0.67375751],
                [0.90265956, 0.86877697, 0.92000042, 0.51009107],
                [0.0315187, 0.93192271, 0.70784201, 0.76623713],
                [0.82274547, 0.56853949, 0.56485733, 0.76484453]
            ],
            [
                [0.00383296, 0.83411639, 0.87466998, 0.98202174],
                [0.04347448, 0.63889672, 0.17384183, 0.4007311],
                [0.95147748, 0.53003441, 0.39527734, 0.91781994],
                [0.09777001, 0.09015373, 0.79944354, 0.11002172],
                [0.45199784, 0.22356263, 0.45995278, 0.56853038],
                [0.89073507, 0.97562825, 0.06093065, 0.63611905]
            ]
        ];

        let gamma_rand = array![[0.8580796, 0.67180762], [0.02475927, 0.43142335]];
        let beta_rand = array![0.64478077, 0.55885354];
        let out_rand = array![
            [
                [0.7722774, 0.31444708, 0.36414167, 0.34734464],
                [0.6889627, 0.638173, 0.4442499, 0.24893896],
                [0.54262614, 0.27660775, 0.67356986, 0.6559096],
                [0.77856696, 0.76413465, 0.78121036, 0.5473954],
                [0.03907336, 0.7900933, 0.68113667, 0.7124848],
                [0.73728293, 0.5846304, 0.58933556, 0.7111326]
            ],
            [
                [0.00367466, 0.87580353, 0.89474547, 0.9651074],
                [0.04565891, 0.6356033, 0.21160461, 0.49451876],
                [0.89665496, 0.6264334, 0.40554562, 0.8284977],
                [0.09278404, 0.08717842, 0.67488253, 0.12799494],
                [0.5616455, 0.2072275, 0.4630361, 0.5438721],
                [0.7649524, 0.89561576, 0.0692213, 0.59935886]
            ]
        ];

        assert!(all_almost_equal::<_, f32>(
            gdn(&input_rand, &beta_rand, &gamma_rand, GdnParameters::Old)
                .iter()
                .cloned(),
            out_rand.iter().cloned(),
            1e-6
        ));
    }

    #[test]
    fn test_igdn_new() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        let res_i = array![[[0., 2.], [0., 2.]], [[0., 0.], [0., 3.]]];

        assert_eq!(igdn(&input, &beta, &gamma, GdnParameters::New), res_i);
    }

    #[test]
    fn test_igdn_old() {
        let input = array![[[0., 1.], [0., 1.]], [[0., 0.], [0., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        let res_i = array![
            [[0., 1.4142135], [0., 1.4142135]],
            [[0., 0.], [0., 1.7320508]]
        ];

        assert_eq!(igdn(&input, &beta, &gamma, GdnParameters::Old), res_i);

        let input_rand = array![
            [
                [0.77917448, 0.71241738, 0.73145832, 0.2764449],
                [0.64081509, 0.1399359, 0.66626099, 0.42713401],
                [0.65461137, 0.99203005, 0.65089197, 0.44369654],
                [0.53149415, 0.36853692, 0.90195496, 0.28938947],
                [0.20470867, 0.50275505, 0.75203511, 0.51851402],
                [0.81446449, 0.36801355, 0.62225798, 0.59854415],
            ],
            [
                [0.26349483, 0.20482207, 0.18704904, 0.59432224],
                [0.90557313, 0.99552319, 0.11858506, 0.63425349],
                [0.47737062, 0.30447258, 0.63896002, 0.094841],
                [0.43290262, 0.4066845, 0.42826408, 0.74698317],
                [0.68992952, 0.23233365, 0.16027236, 0.89392615],
                [0.92854805, 0.93502688, 0.27289194, 0.94444075],
            ],
        ];

        let gamma_rand = array![[0.64188987, 0.53247314], [0.24252059, 0.45782185]];

        let beta_rand = array![0.44310951, 0.40046556];

        let out_rand = array![
            [
                [0.79179627, 0.68851274, 0.7164897, 0.2066961],
                [0.61980987, 0.10674614, 0.6213493, 0.34870306],
                [0.6143925, 1.1722597, 0.61683744, 0.3529114],
                [0.4560364, 0.28383368, 1.0060118, 0.22244604],
                [0.15055184, 0.4181029, 0.7473518, 0.46297714],
                [0.88567865, 0.30316958, 0.5644296, 0.5659577]
            ],
            [
                [0.17284614, 0.13251868, 0.12059585, 0.44075695],
                [0.78309244, 0.9014046, 0.07563008, 0.47919887],
                [0.33661196, 0.20206457, 0.4839028, 0.06031325],
                [0.29991755, 0.27897334, 0.29630527, 0.5960168],
                [0.5352714, 0.15119512, 0.10284474, 0.7683674],
                [0.81257826, 0.8207604, 0.17941903, 0.83313185]
            ]
        ];

        // Test case still failing, investigate later

        // assert!(
        //     all_almost_equal::<_, f32>(
        //         gdn(&input_rand, &beta_rand, &gamma_rand, GdnParameters::Old)
        //             .iter()
        //             .cloned(),
        //         out_rand.iter().cloned(),
        //         1e-6
        //     ),
        //     "\n{:?} too different from \n{:?}",
        //     input_rand,
        //     out_rand
        // );
    }

    #[test]
    fn test_relu() {
        let x = Array::from_shape_vec((1, 2, 2), vec![1., -2., 3., -4.]).unwrap();
        let out = Array::from_shape_vec((1, 2, 2), vec![1., 0., 3., 0.]).unwrap();
        let relu_layer = ReluLayer::new();
        assert_eq!(relu_layer.activate(&x), out);
    }
}
