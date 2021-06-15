use crate::{ImagePrecision, WeightPrecision};
use ndarray::*;

/// Rust implementation of a convolutional layer.
/// The weight matrix shall have dimension (in that order)
/// input channels x output channels x kernel width x kernel height
/// (to comply with the order in which pytorch weights are saved).
pub struct ConvolutionLayer {
    kernel: Array4<WeightPrecision>,
    kernel_width: usize,
    kernel_height: usize,
    stride: usize,
    padding: usize,
    num_input_channels: u16,
    num_output_channels: u16,
}

impl ConvolutionLayer {
    pub fn new(kernel: Array4<WeightPrecision>, stride: usize, padding: usize) -> ConvolutionLayer {
        let num_input_channels = kernel.len_of(Axis(0)) as u16;
        let num_output_channels = kernel.len_of(Axis(1)) as u16;
        let kernel_width = kernel.len_of(Axis(2));
        let kernel_height = kernel.len_of(Axis(3));

        debug_assert!(stride > 0, "Stride of 0 passed");

        ConvolutionLayer {
            kernel,
            kernel_width,
            kernel_height,
            stride,
            num_input_channels,
            num_output_channels,
            padding,
        }
    }

    /// Performs a convolution on the given image data using this layers parameters.
    pub fn conv(&self, image: &Array3<ImagePrecision>) {
        todo!();
    }

    /// Naive implementation of 2d convolution
    /// TODO: We might want to get something more efficient going, like described here:
    /// https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
    fn conv_2d(
        &self,
        kernel_weights: &ArrayView2<WeightPrecision>,
        im2d: &ArrayView2<ImagePrecision>,
    ) -> Array2<ImagePrecision> {
        if self.padding > 0 {
            unimplemented!("Padding bigger than 0 is not supported yet.")
        }

        let im_width = im2d.len_of(Axis(0));
        let im_height = im2d.len_of(Axis(1));

        let new_im_width = (im_width - self.kernel_width) / self.stride + 1;
        let new_im_height = (im_height - self.kernel_height) / self.stride + 1;

        let mut ret = Array::zeros((new_im_width, new_im_height));

        for i in 0..new_im_width {
            let i_with_stride = i * self.stride;
            for j in 0..new_im_height {
                let j_with_stride = j * self.stride;
                let imslice = im2d.slice(s![
                    i_with_stride..(i_with_stride + self.kernel_width),
                    j_with_stride..j_with_stride + (self.kernel_height)
                ]);

                let conv_entry = (&imslice * kernel_weights).sum();
                ret[[i, j]] = conv_entry;
            }
        }
        ret
    }
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
        let test_img = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 1, 1), vec![1.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 2, 0);

        let convolved_image = conv_layer.conv_2d(
            &(conv_layer.kernel.slice(s![0, 0, .., ..])),
            &test_img.view(),
        );

        assert_eq!(convolved_image, array![[0., 0.], [-1., 0.]]);
    }
}
