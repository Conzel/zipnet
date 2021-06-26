use ndarray::*;

use crate::{ImagePrecision, WeightPrecision};

/// Generalized Divisive Normalization activation function.
/// Refer to https://arxiv.org/abs/1912.08771 for a more in-depth explanation.
/// We fix the parameters alpha and epsilon to 1 each.
/// i and j here indicate channel parameters (so the different channels in the image influence each other
/// in the activation)

fn gdn(
    x: &Array3<ImagePrecision>,
    beta: &Array1<WeightPrecision>,
    gamma: &Array2<WeightPrecision>,
) -> Array3<ImagePrecision> {
    let num_channels = x.len_of(Axis(2));

    debug_assert_eq!(gamma.len_of(Axis(0)), num_channels);
    debug_assert_eq!(gamma.len_of(Axis(1)), num_channels);
    debug_assert_eq!(beta.len_of(Axis(0)), num_channels);

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

        // println!("{}", normalization);
        let z_i = &x_i / (beta[i] + normalization);

        // TODO: Same thiing here, a lot of unnecessary assignments :(
        for k in 0..width {
            for l in 0..height {
                z[[k, l, i]] = z_i[[k, l]];
            }
        }
    }
    z
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
}
