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
                GdnParameters::Simplified => {
                    weighted_x_sum = weighted_x_sum + gamma[[j, i]] * x_j.mapv(|a| a.abs())
                }
                GdnParameters::Normal => {
                    weighted_x_sum = weighted_x_sum + gamma[[j, i]] * x_j.mapv(|a| a.powi(2))
                }
            }
        }

        // normalization before applying the epsilon parameter in the exponent
        let normalization_pre_epsilon = beta[i] + weighted_x_sum;

        let normalization = match params {
            GdnParameters::Simplified => normalization_pre_epsilon,
            GdnParameters::Normal => normalization_pre_epsilon.mapv(|a| a.sqrt()),
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
/// Normal: alpha = 2, epsilon = 0.5
/// Simplified: alpha = 1, epsilon = 1
/// Simplified was reported in
/// "Computationally efficient Neural Image Compression", Johnston et al., 2019
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GdnParameters {
    Normal,
    Simplified,
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
    parameters: GdnParameters,
}

impl GdnLayer {
    pub fn new(
        beta: Array1<WeightPrecision>,
        gamma: Array2<WeightPrecision>,
        parameters: GdnParameters,
    ) -> Self {
        Self {
            beta,
            gamma,
            parameters,
        }
    }

    /// Performs GDN on the input with the layer parameters.
    /// We fix the parameter choice to alpha=2, epsilon=0.5.
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        gdn(x, &self.beta, &self.gamma, GdnParameters::Normal)
    }
}

/// Implementation of iGDN as a layer. Refer to the documentation of the free iGDN function
/// for more info.
pub struct IgdnLayer {
    beta: Array1<WeightPrecision>,
    gamma: Array2<WeightPrecision>,
    parameters: GdnParameters,
}

impl IgdnLayer {
    pub fn new(
        beta: Array1<WeightPrecision>,
        gamma: Array2<WeightPrecision>,
        parameters: GdnParameters,
    ) -> Self {
        Self {
            beta,
            gamma,
            parameters,
        }
    }

    /// Performs iGDN on the input with the layer parameters.
    /// We fix the parameter choice to alpha=2, epsilon=0.5.
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        igdn(x, &self.beta, &self.gamma, self.parameters)
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
        assert_eq!(gdn(&input, &beta, &gamma, GdnParameters::Simplified), res);
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
            gdn(&input, &beta, &gamma, GdnParameters::Normal)
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
            gdn(&input_rand, &beta_rand, &gamma_rand, GdnParameters::Normal)
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

        assert_eq!(
            igdn(&input, &beta, &gamma, GdnParameters::Simplified),
            res_i
        );
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

        assert_eq!(igdn(&input, &beta, &gamma, GdnParameters::Normal), res_i);

        let input_rand = array![
            [
                [0.15416284, 0.26331502, 0.01457496, 0.90071485],
                [0.95694934, 0.28382835, 0.94422514, 0.00225923],
                [0.55203763, 0.76813415, 0.76456045, 0.13521018],
                [0.30989758, 0.47122978, 0.28958678, 0.70262236],
                [0.33464753, 0.62458211, 0.76747565, 0.4066403],
                [0.40063163, 0.17756418, 0.41925027, 0.46314887]
            ],
            [
                [0.7400497, 0.53373939, 0.91874701, 0.03342143],
                [0.13720932, 0.60608318, 0.85273554, 0.52122603],
                [0.48537741, 0.16071675, 0.0208098, 0.11627302],
                [0.67145265, 0.8161683, 0.73312598, 0.32756948],
                [0.97805808, 0.95031352, 0.82500925, 0.45130841],
                [0.99513816, 0.9625969, 0.42405245, 0.37372315]
            ]
        ];

        let gamma_rand = array![[0.4655081, 0.03516826], [0.08427267, 0.7325207]];

        let beta_rand = array![0.63619999, 0.02790779];
        let out_rand = array![
            [
                [0.1851324, 0.31642517, 0.01732865, 0.8944951],
                [0.92768925, 0.33811685, 0.89520794, 0.00278283],
                [0.6180026, 0.8038815, 0.80220455, 0.16824557],
                [0.36549708, 0.5282705, 0.34115526, 0.7511118],
                [0.38162732, 0.66060907, 0.78015786, 0.4758259],
                [0.44950372, 0.20797087, 0.48963115, 0.5355754]
            ],
            [
                [1.1286626, 1.0917107, 1.1428818, 0.13967173],
                [0.5047193, 1.1068785, 1.1083646, 1.0941904],
                [1.0561656, 0.6182372, 0.09421822, 0.5929367],
                [1.1166999, 1.127847, 1.1251365, 0.9307215],
                [1.1427177, 1.133284, 1.1152791, 1.0552126],
                [1.1422778, 1.1441946, 1.0413871, 1.0068973]
            ]
        ];

        // Test case still failing, investigate later

        assert!(
            all_almost_equal::<_, f32>(
                gdn(&input_rand, &beta_rand, &gamma_rand, GdnParameters::Normal)
                    .iter()
                    .cloned(),
                out_rand.iter().cloned(),
                1e-6
            ),
            "\n{:?} too different from \n{:?}",
            input_rand,
            out_rand
        );
    }

    #[test]
    fn test_relu() {
        let x = Array::from_shape_vec((1, 2, 2), vec![1., -2., 3., -4.]).unwrap();
        let out = Array::from_shape_vec((1, 2, 2), vec![1., 0., 3., 0.]).unwrap();
        let relu_layer = ReluLayer::new();
        assert_eq!(relu_layer.activate(&x), out);
    }
}
