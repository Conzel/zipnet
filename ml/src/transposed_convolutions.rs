/// Transposed convolutions (also wrongly called deconvolution layers)
/// are learnable upsampling maps.
/// More can be read here:
/// - https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
/// - https://github.com/akutzer/numpy_cnn/blob/master/CNN/Layer/TransposedConv.py
/// - https://ieee.nitk.ac.in/blog/deconv/
use crate::{
    convolutions::{ConvolutionLayer, Padding},
    models::InternalDataRepresentation,
    ConvKernel, ImagePrecision,
};
use ndarray::*;

/// Analog to a Convolution Layer
pub struct TransposedConvolutionLayer {
    convolution_layer: ConvolutionLayer,
}

impl TransposedConvolutionLayer {
    pub fn new(weights: ConvKernel, stride: usize, padding: Padding) -> TransposedConvolutionLayer {
        TransposedConvolutionLayer {
            convolution_layer: ConvolutionLayer::new(weights, stride, padding),
        }
    }

    /// Performs a transposed convolution on the input image. This upsamples the image.
    /// More explanation can be read here:
    /// - https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
    pub fn transposed_convolve(
        &self,
        image: &InternalDataRepresentation,
    ) -> InternalDataRepresentation {
        let output = TransposedConvolutionLayer::ConvTranspose2D(
            self,
            &self.convolution_layer.kernel,
            &image.view(),
        );
        output
    }

    fn ConvTranspose2D<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array3<ImagePrecision>
    where
        // This trait bound ensures that kernel and im2d can be passed as owned array or view.
        // AsArray just ensures that im2d can be converted to an array view via ".into()".
        // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html

        // Input:
        // -----------------------------------------------
        // - x: Input data of shape (C, H, W)
        // - w: Filter weights of shape (F, C, HH, WW)
        // - b: Biases, of shape (F,)
        // -----------------------------------------------
        // - 'stride': The number of pixels between adjacent receptive fields in the
        //     horizontal and vertical directions, must be int
        // - 'pad': "Same" or "Valid"

        // Returns a tuple of:
        // -----------------------------------------------
        // - out: Output data, of shape (F, H', W') where H' and W' are given by
        V: AsArray<'a, ImagePrecision, Ix3>,
        T: AsArray<'a, ImagePrecision, Ix4>,
    {
        // Initialisations
        let im2d_arr: ArrayView3<f32> = im2d.into();
        let kernel_weights_arr: ArrayView4<f32> = kernel_weights.into();
        let output: Array3<ImagePrecision>;
        let new_im_height: usize;
        let new_im_width: usize;
        let filter = self.convolution_layer.num_filters as usize;
        let c_out = self.convolution_layer.img_channels as usize;

        // Dimensions: C, H, W
        let im_channel = im2d_arr.len_of(Axis(0));
        let im_height = im2d_arr.len_of(Axis(1));
        let im_width = im2d_arr.len_of(Axis(2));

        // Calculate output shapes H', W'
        // https://towardsdatascience.com/understand-transposed-convolutions-and-build-your-own-transposed-convolution-layer-from-scratch-4f5d97b2967
        // https://theano-pymc.readthedocs.io/en/latest/tutorial/conv_arithmetic.html
        // H' =  (H - 1) * stride  + HH
        // W' =  (W - 1) * stride  + WW
        new_im_height =
            (im_height - 1) * self.convolution_layer.stride + self.convolution_layer.kernel_height;
        new_im_width =
            (im_width - 1) * self.convolution_layer.stride + self.convolution_layer.kernel_width;

        // weights.reshape(F, HH*WW*C)
        let mut weights_flatten = kernel_weights_arr
            .into_shape(
                filter
                    * self.convolution_layer.kernel_height
                    * self.convolution_layer.kernel_width
                    * c_out,
            )
            .unwrap()
            .to_vec();
        //FLIP
        weights_flatten.reverse();
        let filter_col = Array::from_shape_vec(
            (
                filter,
                self.convolution_layer.kernel_height * self.convolution_layer.kernel_width * c_out,
            ),
            weights_flatten,
        )
        .unwrap();

        // fn:im2col() for with padding always
        // Assuming square kernels:
        let pad_h = self.convolution_layer.kernel_height - 1;
        let pad_w = self.convolution_layer.kernel_width - 1;
        let mut im2d_arr_pad: Array3<ImagePrecision> = Array::zeros((
            im_channel,
            im_height + pad_h + pad_h,
            im_width + pad_w + pad_w,
        ));
        let pad_int_h = im_height + pad_h;
        let pad_int_w = im_width + pad_w;
        // https://github.com/rust-ndarray/ndarray/issues/823
        im2d_arr_pad
            .slice_mut(s![.., pad_h..pad_int_h, pad_w..pad_int_w])
            .assign(&im2d_arr);

        let im_height_pad = im2d_arr_pad.len_of(Axis(1));
        let im_width_pad = im2d_arr_pad.len_of(Axis(2));

        let im_col = self.convolution_layer.im2col_ref(
            im2d_arr_pad.view(),
            self.convolution_layer.kernel_height,
            self.convolution_layer.kernel_width,
            im_height_pad,
            im_width_pad,
            im_channel,
        );

        let filter_transpose = filter_col.t();
        let mul = im_col.dot(&filter_transpose); // + bias_m

        if self.convolution_layer.padding == Padding::Same {
            let mul_reshape = mul
                .into_shape((filter, new_im_height, new_im_width))
                .unwrap()
                .into_owned();
            let (pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right) =
                self.convolution_layer.get_padding_size(
                    im_height,
                    im_width,
                    self.convolution_layer.stride,
                    self.convolution_layer.kernel_height,
                    self.convolution_layer.kernel_width,
                );

            let pad_right_int = new_im_width - pad_right;
            let pad_bottom_int = new_im_height - pad_bottom;
            output = mul_reshape
                .slice(s![.., pad_left..pad_right_int, pad_top..pad_bottom_int])
                .into_owned();
        } else {
            output = mul
                .into_shape((filter, new_im_height, new_im_width))
                .unwrap()
                .into_owned();
        };
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_2d_convT() {
        let test_img = array![[[55.0, 52.0], [57.0, 50.0],]];
        let kernel = Array::from_shape_vec((1, 1, 2, 2), vec![1., 2., 3., 4.]).unwrap();
        let convT_layer = TransposedConvolutionLayer::new(kernel, 1, Padding::Valid);

        let convolved_image = convT_layer.transposed_convolve(&test_img);
        let output1 = arr3(&[[
            [55.0, 162.0, 104.0],
            [222.0, 540.0, 308.0],
            [171.0, 378.0, 200.0],
        ]]);
        assert_eq!(convolved_image, output1);

        let kernel_same = Array::from_shape_vec((1, 1, 2, 2), vec![1., 2., 3., 4.]).unwrap();

        let convT_layer_same = TransposedConvolutionLayer::new(kernel_same, 1, Padding::Same);
        let convolved_image_same = convT_layer_same.transposed_convolve(&test_img);

        let output_same = arr3(&[[[55.0, 162.0], [222.0, 540.0]]]);
        assert_eq!(convolved_image_same, output_same);
    }
}
