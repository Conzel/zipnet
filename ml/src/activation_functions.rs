//! This module provides the necessary activation functions for our neural networks,
//! namely Relu, Generalized Divisive Normalization (GDN) and it's inverse.
//!
//! All activation functions are exposed as a layer as well as a free function
use convolutions_rs::{convolutions::conv2d, Padding};
use ndarray::*;
use num::Float;

use crate::{models::InternalDataRepresentation, ImagePrecision, WeightPrecision};

/// Closely following <https://interdigitalinc.github.io/CompressAI/_modules/compressai/ops/parametrizers.html#NonNegativeParametrizer>
///
/// (i)GDN is reparametrized during training to prevent it from getting stuck in local minima.
/// This is explained in detail in "Efficient Nonlinear Transforms for Lossy Image Compression" by
/// Ballé et al. <https://arxiv.org/pdf/1802.00847.pdf>
///
/// We need to also apply this reparametrization in the GDN layers when importing the
/// weights, else we get wrong results.
struct NonNegativeReparameterizer<F: Float> {
    minimum: F,
    pedestal: F,
}

impl<F: Float> NonNegativeReparameterizer<F> {
    fn new(minimum: F, pedestal: F) -> NonNegativeReparameterizer<F> {
        Self { minimum, pedestal }
    }

    fn reparametrize<D: ndarray::Dimension>(&self, x: &Array<F, D>) -> Array<F, D> {
        x.mapv(|a| a.min(self.minimum).powi(2) - self.pedestal)
    }
}

/// Implementation base for GDN, leveraging that we can use almost the same implementation for GDN and iGDN.
/// The explicit functions (gdn, igdn) are publicly exported to serve a nicer interface.
fn gdn_base(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
    params: GdnParameters,
    inverse: bool,
) -> InternalDataRepresentation {
    // Implementation following:
    // https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/gdn.html#GDN1
    let c = x.shape()[0];
    let gamma_reshape = gamma.to_shape((c, c, 1, 1)).unwrap();
    let mut norm = match params {
        GdnParameters::Simplified => {
            conv2d(&gamma_reshape, x.map(|a| a.abs()).view(), Padding::Valid, 1)
        }
        GdnParameters::Normal => conv2d(
            &gamma_reshape,
            x.map(|a| a.powi(2)).view(),
            Padding::Valid,
            1,
        ),
    };
    // Adding bias (Can be removed once we implement this in conv2d itself)
    for i in 0..c {
        let mut slice = norm.slice_mut(s![i, .., ..]);
        slice += beta[i];
    }
    if !inverse {
        norm = 1.0 / norm;
    }
    if params == GdnParameters::Normal {
        norm = norm.mapv(|a| a.sqrt())
    }
    return x * norm;
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
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        gdn(x, &self.beta, &self.gamma, self.parameters)
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
    use crate::models::CodingModel;

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

    fn get_random_output(params: GdnParameters, inverse: bool) -> Array3<f32> {
        // testing against the implementation:
        // https://interdigitalinc.github.io/CompressAI/layers.html
        // via the following code:
        // from compressai.layers import GDN1
        // import torch
        // torch.manual_seed(0)
        // torch.set_printoptions(sci_mode=False)
        // l = GDN1(10)
        // l.gamma = torch.nn.Parameter(torch.randn(10, 10))
        // l.beta = torch.nn.Parameter(torch.randn(10))
        // gamma_rep = l.gamma_reparam(l.gamma)
        // beta_rep = l.beta_reparam(l.beta)
        // x = torch.randn(1,10,5,6)
        // o = l(x)
        let x = Array::from_shape_vec(
            (10, 5, 6),
            vec![
                1.1404, -0.0899, 0.7298, -1.8453, -0.0250, 1.3694, 2.6570, 0.9851, 0.3772, 1.1012,
                -1.1428, 0.0376, 2.6963, 1.2358, 0.5428, 0.5255, -0.8294, -1.4073, 1.6268, 0.1723,
                -1.6115, -0.4794, -0.1434, -0.3173, 0.5737, 0.9979, 0.5436, 0.0788, 0.8629,
                -0.0195, 0.9910, -0.7777, -0.2994, -0.1878, 1.9159, 0.6902, -2.3217, -1.1964,
                0.1970, -1.1773, 0.1136, 1.1047, -1.3952, 0.4751, -0.8137, 0.9242, -0.2473,
                -1.4154, 0.9874, -1.4878, 0.5867, 0.1583, 0.1102, -0.8188, 0.6328, -1.9169, 1.3119,
                -0.2098, 0.7817, 0.9897, 0.4147, -1.5090, 2.0360, 0.1316, -0.5111, -1.7137,
                -0.5101, -0.4749, -0.6334, -1.4677, -0.8785, -2.0784, -1.1005, -0.7201, 0.0119,
                0.3398, -0.2635, 1.2805, 0.0194, -0.8808, 0.4386, -0.0107, 1.3384, -0.2794,
                -0.5518, -2.8891, -1.5100, 1.0241, 0.1954, -0.7371, 1.7001, 0.3462, 0.9711, 1.4503,
                -0.0519, -0.6284, -0.6538, 1.7198, -0.9610, -0.6375, 0.0747, 0.5600, 0.5314,
                1.2351, -1.1070, -1.7174, 1.5346, -0.0032, -1.6034, 0.0581, -0.6302, 0.7466,
                1.1887, -0.1575, -0.0455, 0.6485, 0.5239, 0.2180, 0.0625, 0.6481, 0.0331, -1.0234,
                0.7335, 1.1177, 2.1494, -0.9088, -0.6710, -0.5804, 1.4903, -0.7005, 0.1806, 1.3615,
                2.0372, 0.6430, -0.7326, -0.4877, 0.2578, -0.5650, 0.9278, 0.4826, -0.8298, 1.2678,
                0.2736, -0.6147, -0.9069, -0.5918, 0.1508, -1.0411, -0.7205, -2.2148, -0.6837,
                0.5164, 0.7928, 0.0832, 0.4228, -1.8687, -1.1057, 0.1437, 0.5836, 1.3482, -1.5771,
                0.3609, -1.3533, -0.2071, -0.2488, -1.2320, 0.6257, -1.2231, -1.1187, 0.3784,
                -0.7804, -0.8739, -0.7328, 0.5143, 0.3976, 0.6435, -1.4453, -0.8078, 1.1975,
                -1.8345, 0.4201, 1.1290, 0.4264, -1.1361, -0.3882, -0.3342, 0.9523, -0.4624,
                -0.6079, -0.3625, -1.5072, -0.5087, 1.1685, 0.7704, 0.3907, 0.2896, -2.7575,
                -0.8324, 0.4900, 0.2908, -1.1311, -0.0009, -0.1627, -0.2477, 2.4197, 1.6456,
                -0.3087, -1.5147, -0.5627, -0.8328, -1.3955, -0.3993, -0.3099, -0.0561, 0.5174,
                -1.5962, 0.3570, -2.2975, -0.8711, -1.6740, 0.5631, -1.4351, 0.7194, -1.3707,
                0.3221, -0.1016, 0.2060, 1.2168, 1.2359, -0.1002, 2.1364, 0.0700, 0.4990, 0.0565,
                0.4061, -1.7384, 1.1901, 2.6352, 0.2284, 0.3241, -1.1154, 2.1914, 0.1158, 0.7773,
                -1.0921, -0.0611, -1.4928, -1.7644, -1.8972, -0.0022, -1.9721, -1.9339, 2.1432,
                -0.9626, -0.5636, 1.6446, 0.2977, -0.6848, -0.0433, 1.8393, -0.5550, 0.7868,
                0.6816, 1.5178, -0.6353, -0.1629, 0.4930, -0.4781, -1.8869, 0.7428, -0.0925,
                -1.4309, -0.5753, -1.4325, -0.6662, 1.0174, -2.2414, 0.4373, -0.5554, -0.0579,
                0.6586, 0.9929, -0.2065, -0.2448, 1.3514, 0.4339, -0.5133, -0.1860, -0.1957,
                0.1610, 0.8669, 0.2292, 0.2294, -0.2544, 1.5800, -0.2444, -0.0282, 0.0513, -0.5151,
                -1.8884,
            ],
        )
        .unwrap();
        let gamma_reparametrized = Array::from_shape_vec(
            (10, 10),
            vec![
                0.0000, 0.0000, 0.0000, 0.0000, 0.7203, 0.4789, 0.0000, 0.0000, 0.1039, 0.0000,
                0.1225, 0.0949, 0.0144, 1.5318, 1.2472, 0.0000, 0.0000, 0.0000, 0.3211, 0.6297,
                0.3586, 0.0000, 0.0000, 3.4336, 0.5628, 0.0000, 0.0000, 0.0337, 1.9303, 2.5165,
                0.8955, 0.0000, 0.0000, 0.0010, 0.0000, 0.0617, 0.1933, 0.0126, 0.4106, 0.1946,
                0.0000, 0.6280, 0.0000, 0.0028, 0.2734, 5.3001, 0.0000, 0.0000, 0.0000, 0.7618,
                1.1138, 0.0316, 0.0000, 0.0000, 0.2952, 0.0000, 0.0000, 0.5536, 2.3134, 11.6315,
                0.0000, 0.0000, 3.3114, 0.0000, 0.0000, 0.8463, 1.2339, 1.6638, 0.0000, 6.5907,
                0.0000, 0.1126, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.2431, 0.0000, 0.6492,
                0.0000, 0.4720, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0784, 0.0054, 1.2395,
                0.0797, 0.1886, 0.2088, 0.0000, 0.6104, 0.0000, 0.0426, 0.0000, 0.0000, 0.0543,
            ],
        )
        .unwrap();
        let beta_reparametrized = Array::from_shape_vec(
            10,
            vec![
                1.7509, 0.0014, 0.0000, 0.0000, 3.5831, 9.6786, 0.0000, 0.0000, 0.0000, 1.5166,
            ],
        )
        .unwrap();
        if inverse {
            IgdnLayer::new(beta_reparametrized, gamma_reparametrized, params).forward_pass(&x)
        } else {
            GdnLayer::new(beta_reparametrized, gamma_reparametrized, params).forward_pass(&x)
        }
    }

    #[test]
    fn test_gdn_simplified_random() {
        let output = Array::from_shape_vec(
            (10, 5, 6),
            vec![
                0.5141, -0.0303, 0.2732, -0.6894, -0.0069, 0.4141, 0.9103, 0.4069, 0.1143, 0.3796,
                -0.4022, 0.0121, 0.6594, 0.5121, 0.2210, 0.1836, -0.3658, -0.5000, 0.5497, 0.0697,
                -0.5797, -0.1515, -0.0605, -0.1222, 0.2156, 0.3988, 0.2089, 0.0268, 0.2839,
                -0.0045, 0.3001, -0.2210, -0.1015, -0.0374, 0.5145, 0.2289, -0.5420, -0.2693,
                0.0451, -0.5459, 0.0731, 0.2802, -0.2999, 0.1412, -0.2149, 0.2201, -0.0768,
                -1.0308, 0.2441, -0.9542, 0.1965, 0.0495, 0.0428, -0.4607, 0.2492, -0.8390, 0.9501,
                -0.1126, 0.3707, 0.1871, 0.0477, -0.1565, 0.3578, 0.0116, -0.0983, -0.2908,
                -0.0421, -0.0425, -0.0677, -0.4609, -0.1405, -0.2314, -0.1277, -0.0952, 0.0013,
                0.0320, -0.0347, 0.4805, 0.0027, -0.1961, 0.0671, -0.0019, 0.2160, -0.0611,
                -0.0906, -0.7398, -0.4807, 0.4090, 0.0333, -0.0762, 1.0367, 0.2286, 1.0230, 0.5854,
                -0.0733, -0.3808, -0.1774, 0.9446, -0.6911, -0.5433, 0.0317, 0.4826, 0.1454,
                0.7140, -1.0283, -1.2578, 0.9960, -0.0018, -0.9467, 0.0569, -0.2953, 0.8756,
                2.2133, -0.1518, -0.0288, 0.4670, 0.6098, 0.3339, 0.0341, 0.6689, 0.0042, -0.1251,
                0.0851, 0.2016, 0.2646, -0.0608, -0.0524, -0.1038, 0.1952, -0.0598, 0.0144, 0.1860,
                0.1650, 0.1203, -0.1102, -0.0437, 0.0343, -0.0502, 0.0880, 0.0712, -0.0924, 0.1435,
                0.0351, -0.0855, -0.1203, -0.0692, 0.0124, -0.1251, -0.0654, -0.1386, -0.0443,
                0.0161, 0.0444, 0.0027, 0.0202, -0.0767, -0.0259, 0.0067, 0.0267, 0.1057, -0.0670,
                0.0134, -0.0642, -0.0121, -0.0090, -0.0637, 0.0358, -0.0770, -0.0753, 0.0233,
                -0.0321, -0.0578, -0.0507, 0.0307, 0.0130, 0.0423, -0.1156, -0.0610, 0.0561,
                -0.0536, 0.0778, 0.0667, 0.0348, -0.0993, -0.0499, -0.0196, 0.0497, -0.0515,
                -0.0719, -0.0377, -0.1324, -0.0304, 0.1329, 0.1040, 0.0384, 0.0521, -0.3233,
                -0.0871, 0.0988, 0.0603, -0.0885, -0.0004, -0.0214, -0.0721, 0.1490, 0.0989,
                -0.0352, -0.1420, -0.0918, -0.0462, -0.7320, -0.2639, -0.3911, -0.0549, 0.4007,
                -0.5863, 0.1652, -0.7016, -0.5944, -0.7436, 0.4938, -0.5621, 0.6068, -0.7153,
                0.2352, -0.1984, 0.3338, 0.6787, 0.6965, -0.2527, 0.6504, 0.2761, 0.6384, 0.1724,
                0.2535, -0.6856, 0.7233, 0.7907, 0.3234, 0.1862, -1.5986, 1.0035, 0.1316, 0.4150,
                -0.6147, -0.0357, -0.3818, -1.3610, -2.2065, -0.0029, -2.1329, -1.0314, 2.1815,
                -1.5032, -0.2700, 1.6597, 0.3863, -0.6864, -0.0537, 2.0003, -0.3647, 2.1350,
                1.7977, 2.1243, -0.2771, -0.1211, 0.6572, -1.2860, -1.8221, 0.2618, -0.0481,
                -0.5230, -0.2253, -0.5651, -0.1988, 0.3716, -0.7880, 0.1887, -0.2071, -0.0225,
                0.3256, 0.3236, -0.0585, -0.1067, 0.5999, 0.2030, -0.2583, -0.0729, -0.0804,
                0.0697, 0.3540, 0.0966, 0.1139, -0.1181, 0.6222, -0.0813, -0.0127, 0.0207, -0.2274,
                -0.5639,
            ],
        )
        .unwrap();
        let our_output = get_random_output(GdnParameters::Simplified, false);
        assert!(
            all_almost_equal::<_, f32>(
                our_output.clone().into_iter(),
                output.clone().into_iter(),
                1e-3
            ),
            "\n{:?} too different from \n{:?}",
            output,
            our_output
        );
    }

    #[test]
    fn test_gdn_simplified_inverse_random() {
        let output = Array::from_shape_vec(
            (10, 5, 6),
            vec![
                2.5292, -0.2663, 1.9492, -4.9391, -0.0904, 4.5281, 7.7552, 2.3850, 1.2450, 3.1950,
                -3.2466, 0.1167, 11.0249, 2.9822, 1.3333, 1.5046, -1.8803, -3.9610, 4.8144, 0.4256,
                -4.4798, -1.5171, -0.3397, -0.8242, 1.5262, 2.4970, 1.4149, 0.2315, 2.6225,
                -0.0838, 3.2734, -2.7371, -0.8833, -0.9416, 7.1348, 2.0813, -9.9452, -5.3153,
                0.8600, -2.5390, 0.1763, 4.3556, -6.4916, 1.5989, -3.0813, 3.8805, -0.7967,
                -1.9435, 3.9942, -2.3198, 1.7513, 0.5061, 0.2838, -1.4554, 1.6068, -4.3796, 1.8115,
                -0.3911, 1.6485, 5.2363, 3.6071, -14.5483, 11.5864, 1.4972, -2.6570, -10.0978,
                -6.1765, -5.3041, -5.9303, -4.6741, -5.4915, -18.6693, -9.4852, -5.4487, 0.1063,
                3.6117, -1.9982, 3.4120, 0.1402, -3.9564, 2.8660, -0.0594, 8.2933, -1.2776,
                -3.3598, -11.2831, -4.7435, 2.5641, 1.1481, -7.1319, 2.7880, 0.5243, 0.9219,
                3.5928, -0.0368, -1.0370, -2.4099, 3.1314, -1.3361, -0.7480, 0.1761, 0.6497,
                1.9416, 2.1364, -1.1918, -2.3448, 2.3643, -0.0059, -2.7156, 0.0594, -1.3450,
                0.6367, 0.6384, -0.1634, -0.0719, 0.9004, 0.4501, 0.1424, 0.1146, 0.6280, 0.2616,
                -8.3709, 6.3191, 6.1956, 17.4581, -13.5830, -8.5853, -3.2461, 11.3755, -8.1999,
                2.2687, 9.9664, 25.1544, 3.4370, -4.8683, -5.4438, 1.9390, -6.3570, 9.7786, 3.2707,
                -7.4487, 11.2044, 2.1308, -4.4166, -6.8378, -5.0584, 1.8292, -8.6629, -7.9336,
                -35.3780, -10.5419, 16.5438, 14.1513, 2.5436, 8.8631, -45.5099, -47.1375, 3.0795,
                12.7666, 17.1957, -37.1291, 9.7362, -28.5357, -3.5401, -6.8972, -23.8332, 10.9235,
                -19.4221, -16.6187, 6.1459, -18.9529, -13.2098, -10.5844, 8.6101, 12.1989, 9.7849,
                -18.0628, -10.6967, 25.5800, -62.8284, 2.2691, 19.1054, 5.2215, -13.0002, -3.0207,
                -5.6913, 18.2524, -4.1473, -5.1376, -3.4855, -17.1533, -8.5193, 10.2703, 5.7044,
                3.9751, 1.6084, -23.5215, -7.9515, 2.4311, 1.4027, -14.4522, -0.0022, -1.2357,
                -0.8514, 39.2927, 27.3908, -2.7074, -16.1572, -3.4495, -14.9944, -2.6605, -0.6041,
                -0.2456, -0.0572, 0.6682, -4.3458, 0.7712, -7.5234, -1.2766, -3.7685, 0.6420,
                -3.6637, 0.8528, -2.6269, 0.4410, -0.0520, 0.1271, 2.1815, 2.1932, -0.0397, 7.0173,
                0.0178, 0.3901, 0.0185, 0.6505, -4.4075, 1.9583, 8.7827, 0.1614, 0.5640, -0.7783,
                4.7854, 0.1018, 1.4557, -1.9401, -0.1046, -5.8368, -2.2873, -1.6313, -0.0017,
                -1.8235, -3.6262, 2.1057, -0.6164, -1.1764, 1.6296, 0.2294, -0.6833, -0.0349,
                1.6913, -0.8447, 0.2899, 0.2584, 1.0844, -1.4562, -0.2191, 0.3698, -0.1778,
                -1.9540, 2.1074, -0.1779, -3.9149, -1.4691, -3.6314, -2.2325, 2.7856, -6.3752,
                1.0136, -1.4894, -0.1494, 1.3321, 3.0468, -0.7289, -0.5616, 3.0447, 0.9274,
                -1.0199, -0.4750, -0.4763, 0.3719, 2.1230, 0.5439, 0.4622, -0.5481, 4.0120,
                -0.7344, -0.0629, 0.1273, -1.1670, -6.3233,
            ],
        )
        .unwrap();
        let our_output = get_random_output(GdnParameters::Simplified, true);

        assert!(
            all_almost_equal::<_, f32>(
                our_output.clone().into_iter(),
                output.clone().into_iter(),
                1e-2
            ),
            "\n{:?} too different from \n{:?}",
            output,
            our_output
        );
    }

    #[test]
    fn test_gdn_normal_random() {
        let output = Array::from_shape_vec(
            (10, 5, 6),
            vec![
                0.7860, -0.0508, 0.4671, -1.1195, -0.0109, 0.6831, 1.5624, 0.6458, 0.1913, 0.6385,
                -0.6226, 0.0200, 1.0922, 0.8398, 0.3660, 0.3070, -0.5871, -0.8492, 0.9439, 0.1127,
                -1.0051, -0.2624, -0.0987, -0.2053, 0.3657, 0.6722, 0.3253, 0.0465, 0.4838,
                -0.0074, 0.4393, -0.3702, -0.1908, -0.0726, 0.7357, 0.4275, -0.9123, -0.4755,
                0.0835, -0.9459, 0.0860, 0.5046, -0.4871, 0.2575, -0.4121, 0.3799, -0.1240,
                -1.3958, 0.4228, -1.1673, 0.3792, 0.0896, 0.0701, -0.7000, 0.3772, -1.4619, 1.5038,
                -0.1701, 0.5375, 0.3248, 0.1156, -0.3841, 0.9499, 0.0334, -0.2080, -0.7522,
                -0.1108, -0.1140, -0.1812, -0.9874, -0.2910, -0.6027, -0.2851, -0.2543, 0.0038,
                0.0848, -0.0868, 0.9212, 0.0060, -0.3390, 0.1901, -0.0052, 0.5492, -0.1257,
                -0.1995, -1.9044, -1.2013, 0.8484, 0.0686, -0.1953, 1.2788, 0.2130, 1.2304, 0.7298,
                -0.0659, -0.4284, -0.2247, 1.1348, -0.7226, -0.5679, 0.0402, 0.4158, 0.1793,
                0.8955, -1.2490, -1.3987, 1.0344, -0.0022, -1.0093, 0.0483, -0.3675, 1.0374,
                2.3213, -0.1518, -0.0315, 0.5291, 0.7196, 0.2661, 0.0410, 0.5764, 0.0128, -0.3808,
                0.2701, 0.4743, 0.7400, -0.1878, -0.1609, -0.2639, 0.5958, -0.1857, 0.0436, 0.5425,
                0.5145, 0.3170, -0.3024, -0.1386, 0.1060, -0.1574, 0.2785, 0.2001, -0.2973, 0.4447,
                0.1073, -0.2607, -0.3478, -0.2063, 0.0380, -0.3836, -0.2088, -0.4336, -0.1753,
                0.0769, 0.2092, 0.0133, 0.0964, -0.3702, -0.1225, 0.0298, 0.1221, 0.3767, -0.3132,
                0.0643, -0.2452, -0.0528, -0.0438, -0.2862, 0.1687, -0.3229, -0.2968, 0.0892,
                -0.1568, -0.2476, -0.2156, 0.1285, 0.0626, 0.1755, -0.4276, -0.2128, 0.2551,
                -0.2497, 0.1985, 0.2349, 0.1039, -0.2915, -0.1816, -0.0684, 0.1580, -0.1390,
                -0.2624, -0.0986, -0.4576, -0.1024, 0.4035, 0.3143, 0.1107, 0.1664, -0.8044,
                -0.2601, 0.2405, 0.1684, -0.2894, -0.0009, -0.0607, -0.2490, 0.4830, 0.2714,
                -0.0899, -0.3529, -0.2980, -0.1542, -0.8761, -0.3161, -0.5281, -0.0484, 0.5088,
                -0.8090, 0.1779, -0.8780, -0.8130, -0.8772, 0.6842, -0.7855, 0.7625, -0.8848,
                0.2732, -0.2112, 0.4289, 0.8420, 0.8667, -0.1899, 0.8580, 0.3376, 0.8495, 0.1621,
                0.2969, -0.8473, 0.8512, 0.8966, 0.4130, 0.2027, -1.4014, 1.2954, 0.1706, 0.4855,
                -0.7185, -0.0467, -0.5034, -1.5224, -2.7401, -0.0024, -2.5688, -1.3745, 2.1047,
                -1.6679, -0.3504, 2.0369, 0.4967, -0.6508, -0.0546, 1.7570, -0.4604, 2.7706,
                2.2353, 2.3723, -0.3498, -0.1139, 0.5124, -0.6335, -2.3576, 0.3358, -0.0680,
                -0.8386, -0.3441, -0.8675, -0.2946, 0.5940, -1.1596, 0.2992, -0.3203, -0.0357,
                0.4747, 0.5064, -0.0896, -0.1702, 0.9330, 0.3162, -0.3651, -0.1149, -0.1250,
                0.1074, 0.5662, 0.1443, 0.1645, -0.1844, 0.9858, -0.1170, -0.0184, 0.0324, -0.3603,
                -0.8418,
            ],
        )
        .unwrap();
        let our_output = get_random_output(GdnParameters::Normal, false);

        assert!(
            all_almost_equal::<_, f32>(
                our_output.clone().into_iter(),
                output.clone().into_iter(),
                1e-3
            ),
            "\n{:?} too different from \n{:?}",
            output,
            our_output
        );
    }

    #[test]
    fn test_gdn_normal_inverse_random() {
        let output = Array::from_shape_vec(
            (10, 5, 6),
            vec![
                1.6544, -0.1591, 1.1402, -3.0416, -0.0575, 2.7451, 4.5186, 1.5027, 0.7437, 1.8993,
                -2.0977, 0.0707, 6.6562, 1.8185, 0.8052, 0.8996, -1.1716, -2.3320, 2.8038, 0.2634,
                -2.5837, -0.8760, -0.2082, -0.4904, 0.8999, 1.4816, 0.9085, 0.1335, 1.5390,
                -0.0514, 2.2359, -1.6339, -0.4699, -0.4856, 4.9894, 1.1143, -5.9086, -3.0102,
                0.4647, -1.4654, 0.1500, 2.4187, -3.9962, 0.8767, -1.6068, 2.2487, -0.4933,
                -1.4352, 2.3062, -1.8963, 0.9078, 0.2797, 0.1733, -0.9577, 1.0616, -2.5134, 1.1445,
                -0.2588, 1.1369, 3.0158, 1.4884, -5.9276, 4.3640, 0.5177, -1.2558, -3.9046,
                -2.3485, -1.9788, -2.2143, -2.1816, -2.6523, -7.1668, -4.2485, -2.0395, 0.0374,
                1.3614, -0.7993, 1.7798, 0.0626, -2.2883, 1.0119, -0.0220, 3.2615, -0.6212,
                -1.5268, -4.3828, -1.8980, 1.2363, 0.5566, -2.7820, 2.2602, 0.5628, 0.7665, 2.8819,
                -0.0409, -0.9219, -1.9025, 2.6064, -1.2779, -0.7156, 0.1387, 0.7540, 1.5753,
                1.7035, -0.9812, -2.1086, 2.2765, -0.0048, -2.5472, 0.0699, -1.0809, 0.5374,
                0.6087, -0.1635, -0.0658, 0.7947, 0.3814, 0.1787, 0.0955, 0.7288, 0.0855, -2.7505,
                1.9917, 2.6338, 6.2431, -4.3967, -2.7990, -1.2766, 3.7279, -2.6421, 0.7470, 3.4171,
                8.0668, 1.3045, -1.7746, -1.7161, 0.6274, -2.0281, 3.0911, 1.1637, -2.3158, 3.6144,
                0.6975, -1.4492, -2.3648, -1.6976, 0.5983, -2.8260, -2.4868, -11.3136, -2.6669,
                3.4649, 3.0053, 0.5205, 1.8540, -9.4341, -9.9823, 0.6931, 2.7884, 4.8249, -7.9423,
                2.0266, -7.4704, -0.8125, -1.4121, -5.3042, 2.3208, -4.6332, -4.2164, 1.6046,
                -3.8836, -3.0841, -2.4903, 2.0577, 2.5263, 2.3586, -4.8851, -3.0664, 5.6211,
                -13.4772, 0.8891, 5.4262, 1.7488, -4.4274, -0.8302, -1.6331, 5.7407, -1.5384,
                -1.4085, -1.3328, -4.9646, -2.5260, 3.3840, 1.8881, 1.3785, 0.5040, -9.4526,
                -2.6638, 0.9982, 0.5022, -4.4213, -0.0009, -0.4359, -0.2465, 12.1214, 9.9763,
                -1.0603, -6.5023, -1.0627, -4.4984, -2.2229, -0.5044, -0.1819, -0.0649, 0.5262,
                -3.1495, 0.7163, -6.0117, -0.9333, -3.1947, 0.4633, -2.6217, 0.6788, -2.1236,
                0.3796, -0.0488, 0.0989, 1.7586, 1.7624, -0.0529, 5.3198, 0.0145, 0.2932, 0.0197,
                0.5554, -3.5664, 1.6640, 7.7456, 0.1264, 0.5181, -0.8878, 3.7070, 0.0785, 1.2446,
                -1.6599, -0.0798, -4.4265, -2.0449, -1.3136, -0.0021, -1.5141, -2.7210, 2.1824,
                -0.5556, -0.9063, 1.3279, 0.1784, -0.7205, -0.0343, 1.9255, -0.6691, 0.2234,
                0.2078, 0.9711, -1.1537, -0.2330, 0.4743, -0.3608, -1.5102, 1.6431, -0.1258,
                -2.4416, -0.9620, -2.3657, -1.5065, 1.7428, -4.3322, 0.6392, -0.9631, -0.0940,
                0.9136, 1.9470, -0.4759, -0.3523, 1.9576, 0.5954, -0.7215, -0.3012, -0.3062,
                0.2412, 1.3273, 0.3641, 0.3199, -0.3511, 2.5324, -0.5105, -0.0434, 0.0812, -0.7365,
                -4.2360,
            ],
        )
        .unwrap();
        let our_output = get_random_output(GdnParameters::Normal, true);

        assert!(
            all_almost_equal::<_, f32>(
                our_output.clone().into_iter(),
                output.clone().into_iter(),
                1e-2
            ),
            "\n{:?} too different from \n{:?}",
            output,
            our_output
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
