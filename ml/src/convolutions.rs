use crate::{ImagePrecision, WeightPrecision};
use ndarray::*;

/// Rust implementation of a convolutional layer.
/// The weight matrix shall have dimension (in that order)
/// input channels x output channels x kernel width x kernel height
/// (to comply with the order in which pytorch weights are saved).
pub struct ConvolutionLayer {
    /// Weight matrix of the kernel
    kernel: Array4<WeightPrecision>,
    kernel_width: usize,
    kernel_height: usize,
    stride: usize,
    padding: usize,
    num_input_channels: u16,
    num_output_channels: u16,
}

impl ConvolutionLayer {
    pub fn new(
        weights: Array4<WeightPrecision>,
        stride: usize,
        padding: usize,
    ) -> ConvolutionLayer {
        let num_input_channels = weights.len_of(Axis(0)) as u16;
        let num_output_channels = weights.len_of(Axis(1)) as u16;
        let kernel_width = weights.len_of(Axis(2));
        let kernel_height = weights.len_of(Axis(3));

        debug_assert!(stride > 0, "Stride of 0 passed");

        ConvolutionLayer {
            kernel: weights,
            kernel_width,
            kernel_height,
            stride,
            num_input_channels,
            num_output_channels,
            padding,
        }
    }

    /// Performs a convolution on the given image data using this layers parameters.
    pub fn convolve(&self, image: &Array3<ImagePrecision>) -> Array3<ImagePrecision> {
        todo!();
    }

    /// Naive implementation of 2d convolution
    /// TODO: We might want to get something more efficient going, like described here:
    /// https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    fn conv_2d<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array2<ImagePrecision>
    where
        // This trait bound ensures that kernel and im2d can be passed as owned array or view.
        // AsArray just ensures that im2d can be converted to an array view via ".into()".
        // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
        V: AsArray<'a, ImagePrecision, Ix2>,
        T: AsArray<'a, ImagePrecision, Ix2>,
    {
        if self.padding > 0 {
            unimplemented!("Padding bigger than 0 is not supported yet.")
        }
        let im2d_arr: ArrayView2<f32> = im2d.into();
        let kernel_weights_arr: ArrayView2<f32> = kernel_weights.into();

        let im_width = im2d_arr.len_of(Axis(0));
        let im_height = im2d_arr.len_of(Axis(1));

        let new_im_width = (im_width - self.kernel_width) / self.stride + 1;
        let new_im_height = (im_height - self.kernel_height) / self.stride + 1;

        let mut ret = Array::zeros((new_im_width, new_im_height));

        for i in 0..new_im_width {
            let i_with_stride = i * self.stride;
            for j in 0..new_im_height {
                let j_with_stride = j * self.stride;
                let imslice = im2d_arr.slice(s![
                    i_with_stride..(i_with_stride + self.kernel_width),
                    j_with_stride..(j_with_stride + self.kernel_height)
                ]);

                let conv_entry = (&imslice * &kernel_weights_arr).sum();
                ret[[i, j]] = conv_entry;
            }
        }
        ret
    }

    // TODO: Refer to
    // https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
    // to speed up everything
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_2dconv() {
        let test_img = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 2, 2), vec![0., 1., -1., 0.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 1, 0);

        let convolved_image = conv_layer.conv_2d(
            &(conv_layer.kernel.slice(s![0, 0, .., ..])),
            &test_img.view(),
        );

        assert_eq!(convolved_image, array![[1., 0.], [1., 0.]]);
    }

    #[test]
    fn test_naive_2d_conv_with_stride() {
        let test_img: Array2<ImagePrecision> = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 1, 1), vec![1.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 2, 0);

        let convolved_image =
            conv_layer.conv_2d(&(conv_layer.kernel.slice(s![0, 0, .., ..])), &test_img);

        assert_eq!(convolved_image, array![[0., 0.], [-1., 0.]]);
    }
}
