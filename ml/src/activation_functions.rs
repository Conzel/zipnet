use ndarray::*;

use crate::{models::InternalDataRepresentation, ImagePrecision, WeightPrecision};

// Implementation base for GDN, leveraging that we can use almost the same implementation for GDN and iGDN.
// The other functions (gdn, igdn) are publicly exported to serve a nicer interface.
fn gdn_base(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
    inverse: bool,
) -> InternalDataRepresentation {
    let num_channels = x.len_of(Axis(2));
    let width = x.len_of(Axis(0));
    let height = x.len_of(Axis(1));

    let mut z: Array3<ImagePrecision> = Array::zeros((width, height, num_channels));

    for i in 0..num_channels {
        let x_i = x.slice(s![.., .., i]);

        let mut normalization: Array2<ImagePrecision> = Array::zeros((width, height));
        for j in 0..num_channels {
            let x_j = x.slice(s![.., .., j]);
            // TODO: Should we run into some performance problems, this is a bit bad,
            // since it copies the array in a loop...
            normalization = normalization + gamma[[i, j]] * x_j.mapv(|a| a.abs());
        }

        let z_i = if inverse {
            &x_i * (beta[i] + normalization)
        } else {
            &x_i / (beta[i] + normalization)
        };

        // TODO: Same thing here, a lot of unnecessary assignments :(
        for k in 0..width {
            for l in 0..height {
                z[[k, l, i]] = z_i[[k, l]];
            }
        }
    }
    z
}

/// Generalized Divisive Normalization activation function.
/// Refer to Ballé et al., 2016, arXiv:1511.06281v4, https://www.cns.nyu.edu/pub/lcv/balle16a-reprint.pdf
/// for a more in-depth explanation.
/// We fix the parameters alpha and epsilon to 1 each,
/// as elaborated in https://arxiv.org/abs/1912.08771.
/// i and j here indicate channel parameters (so the different channels in the image influence each other
/// in the activation)
pub fn gdn(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
) -> InternalDataRepresentation {
    gdn_base(x, beta, gamma, false)
}

/// Inverse Generalized Divisive Normaliazion, computed by the fix-point method mentioned in
/// Ballé et al., 2016, arXiv:1511.06281v4, https://www.cns.nyu.edu/pub/lcv/balle16a-reprint.pdf
/// We only compute one step of the fixed point iteration (should be sufficient for our use cases,
/// and is in line with the architecture described in Minnen et al, 2018 https://arxiv.org/pdf/1809.02736.pdf)

/// Implementation should conform to the tfc implementation for interoperability with trained python layers.
/// Source code: https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py#L31-L461
/// We essentially just replace the division operation with a multiplication operation in the normalization calculation,
/// leveraging the fact, that we only perform one step of iteration.
pub fn igdn(
    x: &InternalDataRepresentation,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
) -> InternalDataRepresentation {
    gdn_base(x, beta, gamma, true)
}

pub fn leaky_relu<D: Dimension>(data: &Array<ImagePrecision, D>) -> Array<ImagePrecision, D> {
    data.mapv(|x| if x > 0. { x } else { 0.01 * x })
}

pub fn relu<D: Dimension>(data: &Array<ImagePrecision, D>) -> Array<ImagePrecision, D> {
    data.mapv(|x| if x > 0. { x } else { 0. })
}

pub struct GdnLayer {
    beta: Array1<WeightPrecision>,
    gamma: Array2<WeightPrecision>,
}

impl GdnLayer {
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        gdn(x, &self.beta, &self.gamma)
    }
}

pub struct IgdnLayer {
    beta: Array1<WeightPrecision>,
    gamma: Array2<WeightPrecision>,
}

impl IgdnLayer {
    pub fn activate(&self, x: &InternalDataRepresentation) -> InternalDataRepresentation {
        igdn(x, &self.beta, &self.gamma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn() {
        let input = array![[[0., 0.], [1., 0.]], [[0., 0.], [1., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        let res = array![[[0., 0.], [0.5, 0.]], [[0., 0.], [(1. / 3.), 0.5]]];
        assert_eq!(gdn(&input, &beta, &gamma), res);
    }

    #[test]
    fn test_igdn() {
        let input = array![[[0., 0.], [1., 0.]], [[0., 0.], [1., 1.]]];
        let beta = array![1., 1.];
        let gamma = array![[1., 1.], [0., 1.]];

        let res_i = array![[[0., 0.], [2., 0.]], [[0., 0.], [3., 2.]]];

        assert_eq!(igdn(&input, &beta, &gamma), res_i);
    }
}
