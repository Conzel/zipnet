use crate::{models::InternalDataRepresentation, ImagePrecision, WeightPrecision};
use ndarray::*;

#[derive(PartialEq, Debug)]
pub enum Padding {
    Same,
    Valid,
}

/// Rust implementation of a convolutional layer.
/// The weight matrix shall have dimension (in that order)
/// input channels x output channels x kernel width x kernel height
/// (to comply with the order in which pytorch weights are saved).
pub struct ConvolutionLayer {
    /// Weight matrix of the kernel
    pub kernel: Array4<WeightPrecision>,
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub stride: usize,
    pub padding: Padding,
    pub num_filters: u16,
    pub img_channels: u16,
}

impl ConvolutionLayer {
    /// Creates new convolution layer. The weights are given in
    /// Pytorch layout.
    /// (out channels, in channels, kernel height, kernel width)
    pub fn new(
        weights: Array4<WeightPrecision>,
        stride: usize,
        padding: Padding,
    ) -> ConvolutionLayer {
        let num_filters = weights.len_of(Axis(0)) as u16; // Filters
        let img_channels = weights.len_of(Axis(1)) as u16; // Channels
        let kernel_height = weights.len_of(Axis(2)); // Height
        let kernel_width = weights.len_of(Axis(3)); // Width

        debug_assert!(stride > 0, "Stride of 0 passed");

        ConvolutionLayer {
            kernel: weights,
            kernel_width,
            kernel_height,
            stride,
            num_filters,
            img_channels,
            padding,
        }
    }

    /// Creates new convolution layer. The weights are given in
    /// Tensorflow layout.
    /// (kernel height, kernel width, in channels, out channels)
    pub fn new_tf(
        weights: Array4<WeightPrecision>,
        stride: usize,
        padding: Padding,
    ) -> ConvolutionLayer {
        let permuted_view = weights.view().permuted_axes([3, 2, 0, 1]);
        // Hack to fix the memory layout, permuted axes makes a
        // col major array / non-contiguous array from weights
        let permuted_array: Array4<WeightPrecision> = Array::from_shape_vec(
            permuted_view.dim(),
            permuted_view.iter().map(|x| *x).collect(),
        )
        .unwrap();
        ConvolutionLayer::new(permuted_array, stride, padding)
    }

    /// Performs a convolution on the given image data using this layers parameters.
    /// We always convolve on flattened images and expect the input array in im2col
    /// style format (read more here).
    /// https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster
    pub fn convolve(&self, image: &InternalDataRepresentation) -> InternalDataRepresentation {
        let output = ConvolutionLayer::Conv2D(self, &self.kernel, &image.view());
        output
    }

    /// Naive implementation of 2d convolution for reference implementations
    fn conv2d_naive<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array2<ImagePrecision>
    where
        // This trait bound ensures that kernel and im2d can be passed as owned array or view.
        // AsArray just ensures that im2d can be converted to an array view via ".into()".
        // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
        V: AsArray<'a, ImagePrecision, Ix2>,
        T: AsArray<'a, ImagePrecision, Ix2>,
    {
        // TODO: Implement valid padding
        if self.padding == Padding::Same {
            unimplemented!("Padding == Same is not implemented for naive conv2d");
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

    pub fn get_padding_size(
        &self,
        input_H: usize,
        input_W: usize,
        stride: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> (usize, usize, usize, usize, usize, usize) {
        let pad_along_height: usize;
        let pad_along_width: usize;
        let idx_0: usize = 0;

        if input_H % stride == idx_0 {
            pad_along_height = (kernel_h - stride).max(idx_0);
        } else {
            pad_along_height = (kernel_h - (input_H % stride)).max(idx_0);
        };
        if input_W % stride == idx_0 {
            pad_along_width = (kernel_w - stride).max(idx_0);
        } else {
            pad_along_width = (kernel_w - (input_W % stride)).max(idx_0);
        };

        let pad_top = pad_along_height / 2;
        let pad_bottom = pad_along_height - pad_top;
        let pad_left = pad_along_width / 2;
        let pad_right = pad_along_width - pad_left;

        (
            pad_along_height,
            pad_along_width,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        )
    }

    pub fn im2col_ref<'a, T>(
        &self,
        im_arr: T,
        ker_height: usize,
        ker_width: usize,
        im_height: usize,
        im_width: usize,
        im_channel: usize,
    ) -> Array2<ImagePrecision>
    where
        // Args:
        //   im_arr: image matrix to be translated into columns, (C,H,W)
        //   ker_height: filter height (hh)
        //   ker_width: filter width (ww)
        //   im_height: image height
        //   im_width: image width
        //
        // Returns:
        //   col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
        //         new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
        T: AsArray<'a, ImagePrecision, Ix3>,
    {
        let im2d_arr: ArrayView3<f32> = im_arr.into();
        let new_h = (im_height - ker_height) / self.stride + 1;
        let new_w = (im_width - ker_width) / self.stride + 1;
        let mut cols_img: Array2<ImagePrecision> =
            Array::zeros((new_h * new_w, im_channel * ker_height * ker_width));
        let mut cont = 0 as usize;
        for i in 1..new_h + 1 {
            for j in 1..new_w + 1 {
                let patch = im2d_arr.slice(s![
                    ..,
                    i - 1 * self.stride..(i - 1 * self.stride + ker_height),
                    j - 1 * self.stride..(j - 1 * self.stride + ker_width),
                ]);
                let patchrow_unwrap: Array1<f32> = Array::from_iter(patch.map(|a| *a));

                cols_img.row_mut(cont).assign(&patchrow_unwrap);
                cont += 1;
            }
        }
        cols_img
    }

    fn col2im_ref<'a, T>(
        &self,
        mat: T,
        height_prime: usize,
        width_prime: usize,
        c: usize,
    ) -> Array3<ImagePrecision>
    where
        T: AsArray<'a, ImagePrecision, Ix2>,
    {
        let img_vec: ArrayView2<f32> = mat.into();
        let filter_axis = img_vec.len_of(Axis(1));
        let mut img_mat: Array3<ImagePrecision> =
            Array::zeros((filter_axis, height_prime, width_prime));
        // C = 1
        for i in 0..filter_axis {
            let col = img_vec.slice(s![.., i]).to_vec();
            let col_reshape = Array::from_shape_vec((height_prime, width_prime), col).unwrap();
            img_mat
                .slice_mut(s![i, 0..height_prime, 0..width_prime])
                .assign(&col_reshape);
        }
        img_mat
    }

    fn Conv2D<'a, T, V>(&self, kernel_weights: T, im2d: V) -> Array3<ImagePrecision>
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
        let im_col: Array2<ImagePrecision>; // output of fn: im2col_ref()
        let new_im_height: usize;
        let new_im_width: usize;
        let filter = self.num_filters as usize;
        let c_out = self.img_channels as usize;

        // Dimensions: C, H, W
        let im_channel = im2d_arr.len_of(Axis(0));
        let im_height = im2d_arr.len_of(Axis(1));
        let im_width = im2d_arr.len_of(Axis(2));

        // Calculate output shapes H', W' for two types of Padding
        if self.padding == Padding::Same {
            // https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
            // H' = H / stride
            // W' = W / stride

            let h_float = im_height as f32;
            let w_float = im_width as f32;
            let stride_float = self.stride as f32;

            let new_im_height_float = (h_float / stride_float).ceil();
            let new_im_width_float = (w_float / stride_float).ceil();

            new_im_height = new_im_height_float as usize;
            new_im_width = new_im_width_float as usize;
        } else {
            // H' =  ((H - HH) / stride ) + 1
            // W' =  ((W - WW) / stride ) + 1
            new_im_height = ((im_height - self.kernel_height) / self.stride) + 1;
            new_im_width = ((im_width - self.kernel_width) / self.stride) + 1;
        };

        // weights.reshape(F, HH*WW*C)
        let filter_col = kernel_weights_arr
            .into_shape((filter, self.kernel_height * self.kernel_width * c_out))
            .unwrap();

        // fn:im2col() for different Paddings
        if self.padding == Padding::Same {
            // https://mmuratarat.github.io/2019-01-17/implementing-padding-schemes-of-tensorflow-in-python
            let (pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right) =
                ConvolutionLayer::get_padding_size(
                    self,
                    im_height,
                    im_width,
                    self.stride,
                    self.kernel_height,
                    self.kernel_width,
                );
            let mut im2d_arr_pad: Array3<ImagePrecision> =
                Array::zeros((c_out, im_height + pad_num_h, im_width + pad_num_w));
            let pad_bottom_int = (im_height + pad_num_h) - pad_bottom;
            let pad_right_int = (im_width + pad_num_w) - pad_right;
            // https://github.com/rust-ndarray/ndarray/issues/823
            im2d_arr_pad
                .slice_mut(s![.., pad_top..pad_bottom_int, pad_left..pad_right_int])
                .assign(&im2d_arr);

            let im_height_pad = im2d_arr_pad.len_of(Axis(1));
            let im_width_pad = im2d_arr_pad.len_of(Axis(2));

            im_col = ConvolutionLayer::im2col_ref(
                self,
                im2d_arr_pad.view(),
                self.kernel_height,
                self.kernel_width,
                im_height_pad,
                im_width_pad,
                im_channel,
            );
        } else {
            im_col = ConvolutionLayer::im2col_ref(
                self,
                im2d_arr,
                self.kernel_height,
                self.kernel_width,
                im_height,
                im_width,
                im_channel,
            );
        };
        let filter_transpose = filter_col.t();
        let mul = im_col.dot(&filter_transpose); // + bias_m
        let activations = ConvolutionLayer::col2im_ref(self, &mul, new_im_height, new_im_width, 1);
        activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_2d_conv() {
        let test_img = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 2, 2), vec![0., 1., -1., 0.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 1, Padding::Valid);

        let convolved_image = conv_layer.conv2d_naive(
            &(conv_layer.kernel.slice(s![0, 0, .., ..])),
            &test_img.view(),
        );

        assert_eq!(convolved_image, array![[1., 0.], [1., 0.]]);
    }

    #[test]
    fn test_naive_2d_conv_with_stride() {
        let test_img: Array2<ImagePrecision> = array![[0., 1., 0.], [0., 0., 0.], [-1., 0., 0.]];
        let kernel = Array::from_shape_vec((1, 1, 1, 1), vec![1.]).unwrap();
        let conv_layer = ConvolutionLayer::new(kernel, 2, Padding::Valid);

        let convolved_image =
            conv_layer.conv2d_naive(&(conv_layer.kernel.slice(s![0, 0, .., ..])), &test_img);

        assert_eq!(convolved_image, array![[0., 0.], [-1., 0.]]);
    }

    #[test]
    fn test_2d_conv() {
        let test_img = array![
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ]
        ];
        let kernel = Array::from_shape_vec(
            (1, 3, 2, 2),
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
        );
        let testker = kernel.unwrap();
        let conv_layer = ConvolutionLayer::new(testker, 1, Padding::Valid);
        let output = arr3(&[[
            [57.0, 75.0, 93.0],
            [111.0, 129.0, 141.0],
            [138.0, 156.0, 162.0],
        ]]);
        let convolved_image = conv_layer.Conv2D(&(conv_layer.kernel), &test_img.view());

        assert_eq!(convolved_image, output);

        let test_img1 = array![
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0, 9.0],
                [7.0, 8.0, 9.0, 9.0]
            ]
        ];
        let kernel1 = Array::from_shape_vec(
            (1, 3, 2, 2),
            vec![1., 2., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.],
        );
        let testker1 = kernel1.unwrap();
        let conv_layer1 = ConvolutionLayer::new(testker1, 1, Padding::Same);
        let output1 = arr3(&[[
            [57.0, 75.0, 93.0, 33.0],
            [111.0, 129.0, 141.0, 48.0],
            [138.0, 156.0, 162.0, 54.0],
            [69.0, 78.0, 81.0, 27.0],
        ]]);
        let convolved_image1 = conv_layer1.Conv2D(&(conv_layer1.kernel), &test_img1.view());

        assert_eq!(convolved_image1, output1);
    }

    #[test]
    fn test_conv2d_tf_layout() {
        let weights_pt = Array::from_shape_vec(
            (2, 1, 3, 3),
            vec![
                0.06664403, 0.65961174, 0.49895822, 0.80375346, 0.20159994, 0.25319365, 0.0520944,
                0.33067411, 0.76843672, 0.08252145, 0.22638044, 0.09291164, 0.63277792, 0.50181511,
                0.40393298, 0.19495441, 0.30511827, 0.28940649,
            ],
        )
        .unwrap();

        let weights_tf = Array::from_shape_vec(
            (3, 3, 1, 2),
            vec![
                0.06664403, 0.08252145, 0.65961174, 0.22638044, 0.49895822, 0.09291164, 0.80375346,
                0.63277792, 0.20159994, 0.50181511, 0.25319365, 0.40393298, 0.0520944, 0.19495441,
                0.33067411, 0.30511827, 0.76843672, 0.28940649,
            ],
        )
        .unwrap();

        let im = array![[
            [0.56494069, 0.3395626, 0.71270928],
            [0.04827336, 0.12623257, 0.30822787],
            [0.82976574, 0.8590054, 0.90254945]
        ]];

        let conv_pt = ConvolutionLayer::new(weights_pt, 1, Padding::Valid);
        let conv_tf = ConvolutionLayer::new_tf(weights_tf, 1, Padding::Valid);
        assert_eq!(conv_pt.convolve(&im), conv_tf.convolve(&im));
    }
}
