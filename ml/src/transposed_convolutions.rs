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
    /// Creates new transposed_convolutionLayer. The weights are given in
    /// Pytorch layout.
    /// (in channels, out channels, kernel_height, kernel_width)
    pub fn new(weights: ConvKernel, stride: usize, padding: Padding) -> TransposedConvolutionLayer {
        TransposedConvolutionLayer {
            convolution_layer: ConvolutionLayer::new(weights, stride, padding),
        }
    }

    /// Creates new transposed_convolutionLayer. The weights are given in
    /// Tensorflow layout.
    /// (kernel height, kernel width, out channels, in channels)
    pub fn new_tf(
        weights: ConvKernel,
        stride: usize,
        padding: Padding,
    ) -> TransposedConvolutionLayer {
        TransposedConvolutionLayer {
            convolution_layer: ConvolutionLayer::new_tf(weights, stride, padding),
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
        // NOTE: THERE IS A CHANGE IN KERNEL DIMENSIONS FOR CONV TRANSPOSED
        // Input:
        // -----------------------------------------------
        // - x: Input data of shape (C, H, W)
        // - w: Filter weights of shape (C, F, HH, WW) // DIFFERENT from CONV2D
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
        let filter = self.convolution_layer.img_channels as usize;
        let c_out = self.convolution_layer.num_filters as usize;

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
        // loop over each filter and flip the kernel and append it back
        let mut filter_col: Array2<ImagePrecision> = Array::zeros((
            c_out,
            self.convolution_layer.kernel_height * self.convolution_layer.kernel_width * filter,
        ));
        for i in 0..c_out {
            let patch_kernel_weights = kernel_weights_arr.slice(s![i, .., .., ..]);
            let mut weights_flatten = patch_kernel_weights
                .into_shape(
                    self.convolution_layer.kernel_height
                        * self.convolution_layer.kernel_width
                        * filter,
                )
                .unwrap()
                .to_vec();
            //FLIP
            weights_flatten.reverse();
            let weights_reverse = Array::from_shape_vec(
                (self.convolution_layer.kernel_height
                    * self.convolution_layer.kernel_width
                    * filter,),
                weights_flatten,
            )
            .unwrap();
            filter_col
                .slice_mut(s![
                    i,
                    0..self.convolution_layer.kernel_height
                        * self.convolution_layer.kernel_width
                        * filter
                ])
                .assign(&weights_reverse);
        }

        let filter_col_flatten = filter_col
            .into_shape((
                filter,
                self.convolution_layer.kernel_height * self.convolution_layer.kernel_width * c_out,
            ))
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

        let filter_transpose = filter_col_flatten.t();
        let mul = im_col.dot(&filter_transpose); // + bias_m
                                                 // println!("{:?}", filter_transpose);

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
                .slice(s![.., pad_top..pad_bottom_int, pad_left..pad_right_int])
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

        let test_img_new = array![
            [
                [0.23224549, 0.50588505, 0.86441349, 0.02310899],
                [0.45685568, 0.40417363, 0.25985479, 0.09913059],
                [0.79699722, 0.98004136, 0.25103959, 0.11597095],
                [0.72586276, 0.09967188, 0.29483115, 0.22645573]
            ],
            [
                [0.16055934, 0.43114743, 0.90784464, 0.96178347],
                [0.63828966, 0.534928, 0.68839463, 0.58409027],
                [0.75128938, 0.66844715, 0.66343357, 0.46953653],
                [0.46234563, 0.26003667, 0.77429137, 0.328285]
            ]
        ];

        let kernel_new = Array::from_shape_vec(
            (2, 1, 4, 4),
            vec![
                0.83035486, 0.49730704, 0.99242497, 0.83261124, 0.8848362, 0.11227968, 0.83485613,
                0.38707261, 0.42852716, 0.33262721, 0.92346432, 0.73501345, 0.24397685, 0.79674084,
                0.95016545, 0.21724486, 0.86324733, 0.1932244, 0.51769137, 0.32076064, 0.96737749,
                0.00598922, 0.39202869, 0.24141203, 0.82792129, 0.69460177, 0.75072335, 0.97536332,
                0.24372894, 0.49899355, 0.31899844, 0.49396161,
            ],
        )
        .unwrap();

        let convT_layer_new = TransposedConvolutionLayer::new(kernel_new, 1, Padding::Valid);
        let convolved_image_new = convT_layer_new.transposed_convolve(&test_img_new);

        let output_new = arr3(&[[
            [0.3314, 0.9388, 2.1500, 2.4249, 2.0847, 1.5318, 0.3277],
            [1.2912, 2.0397, 3.8575, 3.8853, 2.6703, 1.7880, 0.5110],
            [2.5645, 3.6251, 6.0785, 7.3845, 5.5063, 3.6227, 1.3816],
            [3.2539, 4.0704, 7.3212, 8.8666, 6.0085, 3.7842, 1.5748],
            [2.3201, 3.0958, 6.0149, 6.5279, 4.3343, 2.5870, 1.0202],
            [1.0714, 2.2325, 4.3328, 4.6389, 2.8037, 2.0697, 0.7438],
            [0.2898, 0.8967, 1.3070, 1.3203, 1.0215, 0.7664, 0.2114],
        ]]);
        // assert_eq!(convolved_image_new, output_new);
    }
}
